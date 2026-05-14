# tunablefc_sims

Streamlit apps to simulate filter-cavity transmission, reflection, phase, and group delay from user parameters.

## Layout

| Path | Contents |
|------|----------|
| `app_v2.py` | Main Streamlit app (two-mirror and etalon–mirror models) |
| `tunablefc_design.py` | Design helpers used by notebooks / offline analysis |
| `assets/images/` | Cavity schematics and diagrams |
| `notebooks/` | Jupyter notebooks (phase response, locking notes, etc.) |
| `.streamlit/` | Streamlit theme / client defaults |

## Run

From this directory:

```bash
streamlit run app_v2.py
```

or open in browser: https://tunablefcdemo.streamlit.app/

## Requirements

See `requirements.txt`.

## Folder name

The repository is intended to live in a directory named **`tunablefc_sims`**. If yours is still named `tunablefc_demo`, close your editor and any terminals using that folder, then rename the directory to `tunablefc_sims` and reopen the workspace.
