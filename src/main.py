import torch
import numpy as np
import dgl
import argparse
from HGMAE import HGMAE
from datetime import datetime
from utils import meta_edge_reconstruction_loss, target_attribute_restoration_loss, PFP_reconstruction_loss, \
    dynamic_mask_rate, save_model, EarlyStopping
import random
import glob

def setup_argparser():
    parser = argparse.ArgumentParser(description='Heterogenous Graph Transformer Masked AutoEncoder (HGTMAE)')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads in encoder')
    parser.add_argument('--num_out_heads', type=int, default=1, help='Number of attention heads in decoder')
    parser.add_argument('--hidden_units', type=int, default=128, help='Number of hidden units')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of HGT layers')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--leave_unchanged', type=float, default=0.15, help='Fraction of masked node features to leave unchanged')
    parser.add_argument('--replace_rate', type=float, default=0.1, help='Fraction of masked node features to replace with random values')
    parser.add_argument('--MIN_p_a', type=float, default=0.30, help='The minimum mask rate in TAR')
    parser.add_argument('--MAX_p_a', type=float, default=0.50, help='The maximum mask rate in TAR')
    parser.add_argument('--MER_weight', type=float, default=0.5, help='Weight of MER loss')
    parser.add_argument('--TAR_weight', type=float, default=0.5, help='Weight of TAR loss')
    parser.add_argument('--PFP_weight', type=float, default=0.0, help='Weight of PFP loss')
    parser.add_argument('--temporal', type=bool, default=False, help='Whether to take temporal information into account')
    parser.add_argument('--epochs_to_converge', type=int, default=20, help='Number of epochs to converge after dynamic mask rate has saturated')
    parser.add_argument('--patience', type=int, default=2, help='Early stopping patience')
    parser.add_argument('--runname', type=str, default='HGTMAE_toy_run', help='Name of the run')
    parser.add_argument('--num_saves', type=int, default=5, help='Number of checkpoints to save during training')
    parser.add_argument('--batch_path', type=str, default='toy_graphs/batches/2024-08-11_16-15-00', help='Path to save batches')
    return parser.parse_args()


def train(model, train_batches, val_batches, optimizer, args, TAR_mask_rates, device = torch.device('cuda' if torch.cuda.is_available else 'cpu')):
    runname = f"{args.runname}-{datetime.now().strftime('%d-%m-%H-%M')}/"
    early_stopper = EarlyStopping(patience=args.patience, min_delta=0.001, warmup_epochs=args.epochs) # Do not stop before dynamic mask rate has saturated

    all_losses_MER = []
    all_losses_TAR = []
    all_losses_PFP = []
    all_total_losses = []
    all_val_losses_MER = []
    all_val_losses_TAR = []
    all_val_losses_PFP = []
    all_val_total_losses = []

    for epoch in range(args.epochs + args.epochs_to_converge):
        model.train()
        MER_losses = []
        TAR_losses = []
        PFP_losses = []
        total_losses = []

        for batch_idx, batch_dict in enumerate(train_batches):
            batch_subgraph, batch_features, batch_adj, batch_pos = batch_dict.values()
            batch_subgraph = batch_subgraph.to(device)
            batch_features = {ntype: feat.to(device) for (ntype, feat) in batch_features.items()}
            batch_adj = batch_adj.to_dense().to(device)
            batch_pos = batch_pos.to(device)

            optimizer.zero_grad()
            recon_adj, Z, masked_attributes_indices, P_tilde = model(batch_subgraph, batch_features, TAR_mask_rates[epoch])
            MER_loss = meta_edge_reconstruction_loss(recon_adj, batch_adj)
            TAR_loss = target_attribute_restoration_loss(
                node_attributes=batch_features,
                restored_node_attributes=Z,
                masked_node_indices=masked_attributes_indices,
                node_type_list=batch_subgraph.ntypes,
            )
            PFP_loss = PFP_reconstruction_loss(P_tilde, batch_pos)
            total_loss = args.MER_weight * MER_loss + args.TAR_weight * TAR_loss + args.PFP_weight * PFP_loss
            total_loss.backward()
            optimizer.step()

            MER_losses.append(MER_loss.item())
            TAR_losses.append(TAR_loss.item())
            PFP_losses.append(PFP_loss.item())
            total_losses.append(total_loss.item())

        if ((epoch % ((args.epochs + args.epochs_to_converge)// args.num_saves)) == 0) and (epoch != 0):
            print("Saving checkpoint...")
            save_model(
                model,
                runname,
                epoch,
                args,
                all_losses_MER,
                all_losses_TAR,
                all_losses_PFP,
                all_total_losses,
                all_val_losses_MER,
                all_val_losses_TAR,
                all_val_losses_PFP,
                all_val_total_losses,
            )
            print("Checkpoint saved.")

        avg_MER_loss = sum(MER_losses) / len(MER_losses)
        avg_TAR_loss = sum(TAR_losses) / len(TAR_losses)
        avg_PFP_loss = sum(PFP_losses) / len(PFP_losses)
        avg_total_loss = sum(total_losses) / len(total_losses)

        all_losses_MER.append(avg_MER_loss)
        all_losses_TAR.append(avg_TAR_loss)
        all_losses_PFP.append(avg_PFP_loss)
        all_total_losses.append(avg_total_loss)

        print(f"Epoch {epoch+1}/{args.epochs + args.epochs_to_converge} | MER loss: {avg_MER_loss:.5f} | TAR loss: {avg_TAR_loss:.5f} | PFP loss: {avg_PFP_loss:.5f} | Total loss: {avg_total_loss:.5f}")

        model.eval()
        with torch.no_grad():
            val_MER_losses = []
            val_TAR_losses = []
            val_PFP_losses = []
            val_total_losses = []

            for batch_idx, batch_dict in enumerate(val_batches):
                batch_subgraph, batch_features, batch_adj, batch_pos = batch_dict.values()
                batch_subgraph = batch_subgraph.to(device)
                batch_features = {ntype: feat.to(device) for (ntype, feat) in batch_features.items()}
                batch_adj = batch_adj.to_dense().to(device)
                batch_pos = batch_pos.to(device)

                recon_adj, Z, masked_attributes_indices, P_tilde = model(batch_subgraph, batch_features, TAR_mask_rates[epoch])
                MER_loss = meta_edge_reconstruction_loss(recon_adj, batch_adj)
                TAR_loss = target_attribute_restoration_loss(
                    node_attributes=batch_features,
                    restored_node_attributes=Z,
                    masked_node_indices=masked_attributes_indices,
                    node_type_list=batch_subgraph.ntypes,
                )
                PFP_loss = PFP_reconstruction_loss(P_tilde, batch_pos)
                total_loss = args.MER_weight * MER_loss + args.TAR_weight * TAR_loss + args.PFP_weight * PFP_loss

                val_MER_losses.append(MER_loss.item())
                val_TAR_losses.append(TAR_loss.item())
                val_PFP_losses.append(PFP_loss.item())
                val_total_losses.append(total_loss.item())

            avg_val_MER_loss = sum(val_MER_losses) / len(val_MER_losses)
            avg_val_TAR_loss = sum(val_TAR_losses) / len(val_TAR_losses)
            avg_val_PFP_loss = sum(val_PFP_losses) / len(val_PFP_losses)
            avg_val_total_loss = sum(val_total_losses) / len(val_total_losses)

            all_val_losses_MER.append(avg_val_MER_loss)
            all_val_losses_TAR.append(avg_val_TAR_loss)
            all_val_losses_PFP.append(avg_val_PFP_loss)
            all_val_total_losses.append(avg_val_total_loss)

            print(f"Validation | MER loss: {avg_val_MER_loss:.5f} | TAR loss: {avg_val_TAR_loss:.5f} | PFP loss: {avg_val_PFP_loss:.5f} | Total loss: {avg_val_total_loss:.5f}")

        if early_stopper.counter == 1:
            print("Saving model before early stopping...")
            save_model(
                model,
                runname,
                epoch,
                args,
                all_losses_MER,
                all_losses_TAR,
                all_losses_PFP,
                all_total_losses,
                all_val_losses_MER,
                all_val_losses_TAR,
                all_val_losses_PFP,
                all_val_total_losses,
            )
            print("Model saved before early stopping.")
        if early_stopper(avg_val_total_loss):
            print("Early stopping.")
            break

    print("Training finished.")
    print("Saving final model...")
    save_model(
        model,
        runname,
        epoch,
        args,
        all_losses_MER,
        all_losses_TAR,
        all_losses_PFP,
        all_total_losses,
        all_val_losses_MER,
        all_val_losses_TAR,
        all_val_losses_PFP,
        all_val_total_losses,
    )
    print("Final model saved.")

def main():
    args = setup_argparser()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading data...")
    HG = dgl.load_graphs("toy_graphs/imdb.bin")[0][0]
    edge_type_mapping = {can_etype[1]: (can_etype[0], can_etype[2]) for can_etype in HG.canonical_etypes}
    if args.temporal:
        for can_etype in HG.canonical_etypes:
            HG.edges[(can_etype)].data['time'] = HG.edges[(can_etype)].data['time'].type(torch.IntTensor)
    in_dim = {ntype: HG.nodes[ntype].data['feat'].shape[1] for ntype in HG.ntypes}
    out_dim = {ntype: args.hidden_units for ntype in HG.ntypes}
    hidden_dim = {ntype: args.hidden_units for ntype in HG.ntypes}
    ntypes = HG.ntypes
    etypes = HG.etypes
    print("Data loaded.")
    print("Loading batches...")
    batches = []
    for _, batch in enumerate(glob.glob(f"{args.batch_path}/*")):
        batch_dict = torch.load(batch)
        batches.append(batch_dict)
    print("Batches loaded.")

    # Split batches in train and validation
    batches = random.sample(batches, len(batches))
    train_batches = batches[:int(0.8 * len(batches))]
    val_batches = batches[int(0.8 * len(batches)):]
    print(f"Number of training batches: {len(train_batches)}")
    print(f"Number of validation batches: {len(val_batches)}")

    # Get dynamic mask rate for TAR
    TAR_mask_rates = dynamic_mask_rate(args.MIN_p_a, args.MAX_p_a, args.epochs, args.epochs_to_converge)

    print("Initializing model...")
    model = HGMAE(
        n_layers=args.num_layers,
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        n_heads=args.num_heads,
        n_out_heads=args.num_out_heads,
        node_type_list=ntypes,
        edge_type_list=etypes,
        edge_type_mapping=edge_type_mapping,
        leave_unchanged=args.leave_unchanged,
        replace_rate=args.replace_rate,
        temporal=args.temporal,
    )
    model.to(device)
    print("Model initialized.")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    print("Training model...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train(model, train_batches, val_batches, optimizer, args, TAR_mask_rates, device)
    print("Model trained.")

if __name__ == '__main__':
    main()