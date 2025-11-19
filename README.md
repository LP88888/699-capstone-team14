## Repository Structure

```
data/
	encoded/
	normalized/
	raw/
notebooks/
pipeline/
	common/
	config/
	ingrnorm/
	ingredient_ner/
	logs/
	scripts/
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

5. **Run pipelines:**
	Configure settings in the respective config files:
	- **Ingredient normalization pipeline:**
		- Config: `pipeline/config/ingrnorm.yaml`
		- Run: `python pipeline/scripts/run_ingrnorm.py`
	- **NER training:**
		- Config: `pipeline/config/ingredient_ner.yaml`
		- Run: `python pipeline/scripts/run_ingredient_ner.py`
	
	Note: Run the normalization pipeline first to create the artifacts (dedupe map, encoder maps) that the NER training pipeline needs.