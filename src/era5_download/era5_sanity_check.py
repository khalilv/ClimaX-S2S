import os
import xarray as xr
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def handle_corrupted_file(filename, file_path):
    user_input = input(f"Do you want to delete {filename}? (y/n): ").lower()
    if user_input == 'y':
        try:
            # os.remove(file_path)
            logging.info(f"Deleted file: {filename}")
        except Exception as delete_error:
            logging.error(f"Error deleting file {filename}: {str(delete_error)}")
    else:
        logging.info(f"Keeping file: {filename}")
    logging.info(f"--------------------------------------------------")

def validate_dataset(ds, filename, exp):
    assert 'latitude' in ds.dims and 'longitude' in ds.dims, "Missing latitude or longitude dimensions"
    assert ds.dims['latitude'] == exp['latitude'], f"Unexpected latitude. Expected {exp['latitude']} received {ds.dims['latitude']}"
    assert ds.dims['longitude'] == exp['longitude'], f"Unexpected longitude. Expected {exp['longitude']} received {ds.dims['longitude']}"
    
    if 'valid_time' in ds.dims:
        assert ds.dims['valid_time'] == exp['time'], f"Unexpected temporal dimension. Expected {exp['time']} received {ds.dims['valid_time']}"
    elif 'time' in ds.dims:
        assert ds.dims['time'] == exp['time'], f"Unexpected temporal dimension. Expected {exp['time']} received {ds.dims['time']}"
    else:
        raise AssertionError('Missing temporal dimension (valid_time or time)')
    logging.info(f"{filename} passed all assertions.")

data_dir = '/u/gracefo-m0/DATA_SERVER/s2s.dir/era5/q50' 
expected_values = {'latitude': 721, 'longitude': 1440, 'time': 8760}

for filename in os.listdir(data_dir):
    if filename.endswith('.nc'):
        file_path = os.path.join(data_dir, filename)
        try:
            with xr.open_dataset(file_path) as ds:
                validate_dataset(ds, filename, expected_values)
        except AssertionError as e:
            logging.warning(f"Assertion failed for {filename}: {str(e)}")
            handle_corrupted_file(filename, file_path)
        except Exception as e:
            logging.error(f"Error opening file {filename}: {str(e)}")
            handle_corrupted_file(filename, file_path)