from pathlib import Path
from src.utils.config_loader import load_config
import pickle
import numpy as np
import pandas as pd
import pytest
from src.utils.hyperparam_tuning import update_hyperparameters, load_hyperparameters
import src.utils.graph_utils as gu
from src.evaluation.metrics import calculate_ace, structural_hamming_distance, intervene_and_resample_model
import src.data_processing.data_loader as dl
from src.data_processing.preprocessor import preprocess_data

def test_load_config(tmp_path, monkeypatch):
    # Set up a fake project with config/paths.yaml
    project = tmp_path / 'myproj'
    cfg = project / 'config'
    cfg.mkdir(parents=True)
    paths_yaml = cfg / 'paths.yaml'
    paths_yaml.write_text("""
        data: 
            raw: data/raw
        graphs:
            foo: graphs/foo
        output:
            bar: output/bar
        """)
    # Go into a nested working dir
    (project / 'work' / 'here').mkdir(parents=True)
    monkeypatch.chdir(project / 'work' / 'here')

    config = load_config()
    # All three categories should be converted to Path
    assert isinstance(config['data']['raw'], Path)
    assert config['data']['raw'] == Path('data/raw')
    assert isinstance(config['graphs']['foo'], Path)
    assert config['graphs']['foo'] == Path('graphs/foo')
    assert isinstance(config['output']['bar'], Path)
    assert config['output']['bar'] == Path('output/bar')
    

def test_topological_sort_chain():
    # 0→1→2
    mat = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ])
    order = gu.topological_sort(mat)
    assert order == [0, 1, 2]

def test_remove_bidirectional_edges(monkeypatch):
    # set up a 2-node bidirectional graph
    df = pd.DataFrame(
        [[0, 1],
         [1, 0]],
        index=[0, 1],
        columns=[0, 1],
    )
    # force random.random() always < 0.5
    monkeypatch.setattr("random.random", lambda: 0.3)
    out = gu.remove_bidirectional_edges(df)
    # ensure no bidirectional pairs remain
    assert not ((out.loc[0, 1] != 0) and (out.loc[1, 0] != 0))
    # one of the two edges must survive
    assert (out.loc[0, 1] + out.loc[1, 0]) == 1
    
def test_get_dag(monkeypatch, tmp_path):
    # prepare dict with 'model' and numpy adjacency
    adj_np = pd.DataFrame([[0, 2], [0, 0]], index=["x", "y"], columns=["x", "y"])
    data = {"model": "my_model", "adjmat": adj_np}
    data_folder = tmp_path / "gt2"
    data_folder.mkdir()
    pkl_file = data_folder / "bar.pkl"
    with open(pkl_file, "wb") as f:
        pickle.dump(data, f)

    monkeypatch.setattr(
        gu, "load_config",
        lambda: {"ground_truth": str(data_folder)}
    )
    model, adj = gu.get_dag("bar")
    assert model == "my_model"
    # values cast to int
    assert adj.loc["x", "y"] == 2
    assert adj.dtypes["x"] == int 
    
def test_update_hyperparam_and_loads(tmp_path):
    fp = tmp_path / "hp.yaml"
    # First update on non-existent file
    update_hyperparameters("model1", "data1", {"lr": 0.01}, file_path=str(fp))
    assert fp.exists()

    loaded = load_hyperparameters("model1", "data1", file_path=str(fp))
    assert loaded == {"lr": 0.01}
    

def test_update_overwrites_dataset(tmp_path):
    fp = tmp_path / "hp.yaml"
    update_hyperparameters("modelX", "ds", {"alpha": 0.1}, file_path=str(fp))
    update_hyperparameters("modelX", "ds", {"alpha": 0.2, "beta": 3}, file_path=str(fp))

    loaded = load_hyperparameters("modelX", "ds", file_path=str(fp))
    assert loaded == {"alpha": 0.2, "beta": 3}
    
class DummyModel:
    """A dummy model that, upon intervention at node X, predicts
    each node Y as data[X] + offset[Y]."""

    def __init__(self, offsets):
        self.offsets = offsets

    def predict(self, data, starting_node):
        base = data[starting_node].iloc[0]
        return {node: base + self.offsets.get(node, 0) for node in data.columns if node != starting_node}

def test_structural_hamming_distance_extra_and_missing():
    idx = cols = ['X', 'Y', 'Z']
    true = pd.DataFrame(np.zeros((3, 3), dtype=int), index=idx, columns=cols)
    learned = true.copy()
    # true has X→Y, Y→Z
    true.loc['X', 'Y'] = 1
    true.loc['Y', 'Z'] = 1
    # learned has X→Y (correct), Z→Y (extra), missing Y→Z
    learned.loc['X', 'Y'] = 1
    learned.loc['Z', 'Y'] = 1

    res = structural_hamming_distance(true, learned)
    assert res["extra_edges"] == 1
    assert res["missing_edges"] == 1
    assert res["directed_shd"] == 2
    
def test_intervene_and_resample_model_updates_descendants_and_preserves_others():
    # initial data: two rows identical
    df = pd.DataFrame({
        'A': [0, 0],
        'B': [10, 10],
        'C': [20, 20],
    })
    # offsets: B gets +5, C gets +0
    model = DummyModel(offsets={'B': 5})
    out = intervene_and_resample_model(df.copy(), model, 'A', intervention_value=7)
    assert (out['A'] == 7).all()
    assert (out['B'] == 12).all()
    assert (out['C'] == 7).all()
    
def test_calculate_ace_linear_effect():
    # Data with three distinct intervention classes: 0,1,2
    df = pd.DataFrame({
        'X': [0, 1, 2],
        'Y': [0, 0, 0],
    })
    # model: effect_node Y = X + 3 (via offset)
    model = DummyModel(offsets={'Y': 3})
    # ACE = mean[(3+c - 3*0) for c=1,2] = mean([1,2]) = 1.5
    ace = calculate_ace(model, df, intervention_node='X', effect_node='Y')
    assert pytest.approx(ace) == 1.5


def test_calculate_ace_unsorted_classes():
    # Classes [2, 0, 1] out of order, but function should sort to [0,1,2]
    df = pd.DataFrame({
        'I': [2, 0, 1],
        'E': [0, 0, 0],
    })
    # effect E = I + 2
    model = DummyModel(offsets={'E': 2})
    ace = calculate_ace(model, df, intervention_node='I', effect_node='E')
    # same classes, so same ACE: mean([2+0, 1+0]) = mean([1,2]) = 1.5
    assert pytest.approx(ace) == 1.5
    
class DummyConfig:
    """ Helper to monkey-patch load_config """
    def __init__(self, base_path):
        self.cfg = {
            'data': {
                'synthetic': {
                    'asia': str(base_path / 'asia'),
                }
            }
        }
    def __call__(self):
        return self.cfg
    
@pytest.fixture(autouse=True)
def patch_load_config(monkeypatch, tmp_path):
    """Monkey-patch dl.load_config to return a dummy config rooted at tmp_path."""
    monkeypatch.setattr(dl, 'load_config', DummyConfig(tmp_path))
    return tmp_path

def make_csv(path, df):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def test_load_synthetic_data(patch_load_config):
    tmp_path = patch_load_config
    # create asia/base folder and a single CSV
    folder = tmp_path / 'asia' / 'base'
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    csv = folder / 'data.csv'
    make_csv(csv, df)

    out = dl.load_synthetic_data(dataset='asia', variant='base')
    # should return the single DataFrame
    pd.testing.assert_frame_equal(out, df)

@pytest.fixture
def df_binary():
    # 10 rows, two features and a balanced binary target
    return pd.DataFrame({
        'f1': range(10),
        'f2': range(10, 20),
        'target': [0]*5 + [1]*5
    })


def test_preprocess_data_with_target(df_binary):
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(
        df_binary,
        target_node='target',
        test_size=0.3,
        val_size=0.2,
        random_state=0
    )
    # Sizes: total=10, train=5, val=2, test=3
    assert len(X_train) == 5
    assert len(X_val) == 2
    assert len(X_test) == 3
    # Stratification preserved roughly: each split has both classes if possible
    assert set(y_train.unique()) == {0, 1}
    assert set(y_val.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}
    # Reproducibility
    X_train2, y_train2, *_ = preprocess_data(
        df_binary,
        target_node='target',
        test_size=0.3,
        val_size=0.2,
        random_state=0
    )
    pd.testing.assert_frame_equal(X_train, X_train2)
    pd.testing.assert_series_equal(y_train, y_train2)

def test_preprocess_data_without_target():
    # 12 rows with three columns
    df = pd.DataFrame(np.arange(36).reshape(12,3), columns=list('ABC'))
    train, val, test = preprocess_data(
        df,
        target_node=None,
        test_size=0.25,
        val_size=0.25,
        random_state=1
    )
    # test_size+val_size=0.5 → train=6, then val=3, test=3
    assert len(train) == 6
    assert len(val) == 3
    assert len(test) == 3
    # No overlap
    assert set(train.index).isdisjoint(val.index)
    assert set(train.index).isdisjoint(test.index)
    assert set(val.index).isdisjoint(test.index)
