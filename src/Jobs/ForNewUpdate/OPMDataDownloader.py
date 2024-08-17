import requests
import pandas as pd

class OPMDataDownloader:
    def __init__(self, path):
        self.path = path
        
        self.headers = {
            "authority": "opm-back.cc.lehigh.edu:3000",
            "method": "GET",
            "scheme": "https",
            "Accept": "application/json, text/plain, */*",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
            "Origin": "https://opm.phar.umich.edu",
            "Referer": "https://opm.phar.umich.edu/",
            "Sec-Ch-Ua": '"Not A;Brand";v="99", "Google Chrome";v="91", "Chromium";v="91"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "cross-site",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        }
        self.host = f"https://opm-back.cc.lehigh.edu/opm-backend/{self.path}"

    def fetch_opm_data(self):
        try:
            response = requests.get(self.host, headers=self.headers)
            if response.status_code == 200:
                data = response.json()
                objects = data.get('objects', [])
                opm_df = pd.DataFrame(objects)
                return opm_df
            else:
                print(f"Failed to fetch data. Status code: {response.status_code}")
        except Exception as e:
            print(f"Exception occurred: {str(e)}")
        
        return None

    def save_data_to_csv(self, opm_df, filename):
        if opm_df is not None:
            opm_df.to_csv(filename, index=False)
            print(f"Data saved successfully as {filename}")
        else:
            print("No data to save.")

# Example usage
if __name__ == "__main__":
    path = "/primary_structures?search=&sort=&pageSize=8915"
    downloader = OPMDataDownloader(path)
    opm_df = downloader.fetch_opm_data()

    if opm_df is not None:
        downloader.save_data_to_csv(opm_df, "./datasets/full_opm_data.csv")
    else:
        print("No data fetched.")
