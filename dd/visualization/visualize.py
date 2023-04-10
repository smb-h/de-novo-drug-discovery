import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from sklearn.decomposition import PCA


def mols_to_smiles(smiles):
    # TODO: check smiles validity (remove none from list)
    rs = []
    for smile in smiles:
        rs.append(Chem.MolFromSmiles(smile))
    return rs


def get_smile_fingerprints(smiles):
    mols = mols_to_smiles(smiles)
    mols = [x for x in mols if x is not None]
    fps = []
    for mol in mols:
        bv = AllChem.GetMACCSKeysFingerprint(mol)
        fp = np.zeros((len(mols), len(bv)))
        DataStructs.ConvertToNumpyArray(bv, fp)
        fps.append(fp)
    return fps


def plot_scatter_org_vs_pred(config, model_name, y_org, y_pred):
    y_org_fps = get_smile_fingerprints(y_org)
    y_pred_fps = get_smile_fingerprints(y_pred)

    pca = PCA(n_components=2, random_state=42)
    y_pred_len = len(y_pred_fps)
    x = y_pred_fps + y_org_fps
    x = pca.fit_transform(x)

    # scatter plot of org vs pred
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x[:y_pred_len, 0],
            y=x[:y_pred_len, 1],
            mode="markers",
            name="Predicted",
            marker=dict(color="red"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x[y_pred_len:, 0],
            y=x[y_pred_len:, 1],
            mode="markers",
            name="Original",
            marker=dict(color="blue"),
        )
    )
    fig.update_layout(
        title="Scatter plot of original vs predicted SMILES",
        xaxis_title="PC1",
        yaxis_title="PC2",
        # legend_title="Legend Title",
    )

    # save plot in model logs directory
    model_logs_path = config[f"model_{model_name}"].get("logs_path")
    fig.write_image(f"{model_logs_path}/scatter.png")
    fig.write_html(f"{model_logs_path}/scatter.html")
    print(f"Scatter plot saved in {model_logs_path}")


def plot_violin_org_vs_pred(config, model_name, y_org, y_pred):
    y_pred_mols = mols_to_smiles(y_pred)
    y_pred_mols = [x for x in y_pred_mols if x is not None]
    y_org_mols = mols_to_smiles(y_org)

    props = {
        "MW": {
            "predicted": [Descriptors.MolWt(mol) for mol in y_pred_mols],
            "original": [Descriptors.MolWt(mol) for mol in y_org_mols],
        },
        "LogP": {
            "predicted": [Descriptors.MolLogP(mol) for mol in y_pred_mols],
            "original": [Descriptors.MolLogP(mol) for mol in y_org_mols],
        },
    }

    # violin plot of org vs pred
    fig_mw = go.Figure()
    fig_mw.add_trace(
        go.Violin(
            x=["original"] * len(props["MW"]["original"]),
            y=props["MW"]["original"],
            name="original",
            box_visible=True,
            meanline_visible=True,
        )
    )
    fig_mw.add_trace(
        go.Violin(
            x=["predicted"] * len(props["MW"]["predicted"]),
            y=props["MW"]["predicted"],
            name="predicted",
            box_visible=True,
            meanline_visible=True,
        )
    )
    fig_mw.update_layout(
        title="MW predicted vs original",
        xaxis_title="",
        yaxis_title="",
        legend_title="",
    )

    fig_logp = go.Figure()
    fig_logp.add_trace(
        go.Violin(
            x=["original"] * len(props["LogP"]["original"]),
            y=props["LogP"]["original"],
            name="original",
            box_visible=True,
            meanline_visible=True,
        )
    )
    fig_logp.add_trace(
        go.Violin(
            x=["predicted"] * len(props["LogP"]["predicted"]),
            y=props["LogP"]["predicted"],
            name="predicted",
            box_visible=True,
            meanline_visible=True,
        )
    )
    fig_logp.update_layout(
        title="LogP predicted vs original",
        xaxis_title="",
        yaxis_title="",
        legend_title="",
    )

    model_logs_path = config[f"model_{model_name}"].get("logs_path")
    fig_mw.write_image(f"{model_logs_path}/violin_mw.png")
    fig_mw.write_html(f"{model_logs_path}/violin_mw.html")
    fig_logp.write_image(f"{model_logs_path}/violin_logp.png")
    fig_logp.write_html(f"{model_logs_path}/violin_logp.html")
    print(f"Violin plot saved in {model_logs_path}")
