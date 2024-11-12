import time
import os
import functions_preprocessing as fp

# Range members and years
range_members = [2,50]
range_years = [1955,2099]
range_months = [1,12]
# Variable selection
var = 'zg'
DESIRED_LEVEL = 50000.0  # Replace with your specific pressure level

# Functions
def check_all_files_downloaded(list_path_files):
    """Check if all files in EXPECTED_FILES are downloaded."""
    watch_directory = os.path.dirname(list_path_files[0])
    current_files = set([os.path.join(watch_directory, filename) for filename in os.listdir(watch_directory)])
    expected_files_set = set(list_path_files)
    return expected_files_set.issubset(current_files)

def process_nc_file(path_file):
    """Extract DESIRED_LEVEL from multiple-level .nc file"""
    # Replace 'zg' with 'zg500' in the output filename
    watch_directory = os.path.dirname(path_file)
    base_filename = os.path.basename(path_file).replace('zg', 'zg500')
    output_path = os.path.join(watch_directory, base_filename)
    # print(f"Processing file: {path_file}")
    # Using NCO to extract specific pressure level
    os.system(f"ncks -F -d plev,{DESIRED_LEVEL} {path_file} {output_path}")
    # print(f"Saved processed file to: {output_path}")

def process_all_files(list_path_files):
    """Process all expected .nc files in the directory."""
    for pathf in list_path_files:
        process_nc_file(pathf)

# Main code
if __name__ == "__main__":

    for memb in range(range_members[0], range_members[1] + 1):
        for year in range(range_years[0], range_years[1] + 1):
            # Generate list strings
            files_3h = [
                fp.path_file_CRCM5(var, memb, year, mon, time_res='3h')
                for mon in range(range_months[0], range_months[1] + 1)
            ]
            basedir = os.path.dirname(files_3h[0])
            files_3h_proc = [
                os.path.join(basedir, os.path.basename(files_3h[im]).replace('zg', 'zg500'))
                for im in range(range_months[1] - range_months[0] + 1) 
            ]
            # Wait until all expected files are downloaded
            print(f"Waiting for files member {memb}, year {year}...")
            while not check_all_files_downloaded(files_3h):
                time.sleep(100)
            
            # Process all files in directory once they're all downloaded
            process_all_files(files_3h)
            print("Processing complete on member", memb, "and year", year)

            # Remove original 3h files
            if fp.remove_list2_if_list1_exists(files_3h_proc, files_3h):
                print(f'3h files memb {memb} year {year} removed successfully')
            else:
                print(f'3h files memb {memb} year {year} not removed because zg500 files are missing')
