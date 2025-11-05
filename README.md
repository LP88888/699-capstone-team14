## Repository Structure

```
data/
	encoded/
	normalized/
	raw/
notebooks/
pipeline/
	common
	config/
	ingrnorm/ cosdedupe.py encoder.py multidataset.py parquet_utils.py spellmap.py stats_normalizer.py
	logs/
	run_ingnorm.py
.gitignore
LICENSE
README.md
requirements.txt
```


## Setup Instructions

1. **Create a virtual environment:**
	```sh
	python -m venv venv
	```

2. **Activate the virtual environment:**
	- **Windows:**
	  ```sh
	  venv\Scripts\activate
	  ```
	- **Mac/Linux:**
	  ```sh
	  source venv/bin/activate
	  ```

3. **Install dependencies:**
	```sh
	pip install -r requirements.txt
	```


## Environment Variables (.env)

For sensitive information (API keys, database credentials, etc.), create a `.env` file in the root directory.


4. **Add new libraries:**
	 - When introducing new libraries, add them to `requirements.txt`.
	 - To list installed packages:
		 ```sh
		 pip freeze
		 ```
	 - Copy any new packages to `requirements.txt`.

5. **Run cleaning pipeline:**
	Configure settings in config/config.yaml then run in CLI:
	```sh
	python run_ingrnorm.py
	```