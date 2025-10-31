## Repository Structure

```
data/
notebooks/
pipeline/
	config/
	logs/
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

