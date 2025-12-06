"""Data validation tests."""

def test_dvc_setup():
    """Test DVC setup files exist."""
    import os
    # Check DVC is initialized
    assert os.path.exists('.dvc'), "DVC not initialized"
    assert os.path.exists('.dvc/config'), "DVC config missing"
    
def test_data_files_tracked():
    """Test that data files are tracked by DVC."""
    import os
    import glob
    # Check for .dvc files
    dvc_files = glob.glob('data/**/*.dvc', recursive=True)
    assert len(dvc_files) > 0, "No data files tracked by DVC"
    
def test_notebooks_exist():
    """Test that notebooks exist."""
    import os
    assert os.path.exists('notebooks/01_eda_initial.ipynb'), "EDA notebook missing"
