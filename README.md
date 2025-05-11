# AI Fashion Assistant

The AI Fashion Assistant is a Streamlit web application that helps users manage their virtual wardrobe, generate outfit recommendations using AI, save their favorite outfits, and browse a thrift marketplace for new clothing items.

## Features

- **Virtual Wardrobe:** Upload images of your clothing items, categorize them (top, bottom), and add descriptions.
- **Outfit Generator:** Get AI-powered outfit recommendations based on a text prompt (e.g., "casual summer outfit"). It can also suggest outfits combining your wardrobe items with items from the marketplace.
- **Saved Outfits:** Save your favorite generated outfits for future reference.
- **Thrift Marketplace:** Browse and view details of clothing items available in a simulated thrift marketplace. Get outfit suggestions by combining marketplace items with your existing wardrobe.

## Project Structure

```
.
├── .env                  # Local environment variables (API keys, DB config)
├── .env.example          # Example environment variables file
├── .gitignore            # Specifies intentionally untracked files that Git should ignore
├── app.py                # Main Streamlit application file
├── database.py           # SQLite database interactions (saving/loading outfits)
├── embeddings.py         # Functions for generating image and text embeddings using CLIP
├── fashion.db            # SQLite database file
├── requirements.txt      # Python dependencies
├── vector_db.py          # Qdrant vector database interactions
├── clothes-images/       # Directory for user-uploaded wardrobe images (example)
├── marketplace-images/   # Directory for marketplace item images (example)
└── model/                # Directory for the pre-trained CLIP model
```

## Setup

### 1. Clone the Repository (if applicable)

```bash
git clone <your-repository-url>
cd <your-repository-directory>
```

### 2. Create a Virtual Environment

It's recommended to use a virtual environment to manage project dependencies.

```bash
python -m venv venv
```

Activate the virtual environment:

- On Windows:
  ```bash
  .\venv\Scripts\activate
  ```
- On macOS/Linux:
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies

Install the required Python packages using the [requirements.txt](requirements.txt) file:

```bash
pip install -r requirements.txt
```

### 4. Set up Environment Variables

Create a `.env` file in the root of the project directory by copying the [`.env.example`](.env.example) file:

```bash
cp .env.example .env
```

Edit the `.env` file and add your Qdrant API key and host URL:

```env
QDRANT_THRIFT_API_KEY='your_qdrant_api_key'
QDRANT_HOST='your_qdrant_endpoint_url'
QDRANT_WARDROBE_COLLECTION='clothes'
QDRANT_MARKETPLACE_COLLECTION='marketplace'
CLOTHING_TAGS = ["t-shirt", "shirt", ...] # Keep or modify as needed
```

### 5. Download the CLIP Model

The application uses a pre-trained CLIP model. You will need to download the model files and place them in a `model/` directory in the root of your project. The [`app.py`](app.py) file expects the model to be at `./model`.

You can typically download models from the Hugging Face Model Hub. For example, if you are using `openai/clip-vit-base-patch32`, ensure the `model/` directory contains files like `config.json`, `pytorch_model.bin`, `preprocessor_config.json`, etc.

### 6. Prepare Image Directories

- Create a directory named `images` (or `clothes-images` as per your `.gitignore`) in the root of your project. This is where images uploaded to the virtual wardrobe will be saved by default (see `show_wardrobe_page` function in [`app.py`](app.py)).
- If you have pre-existing marketplace images, ensure they are in a directory (e.g., `marketplace-images/`) and the paths in your Qdrant marketplace collection point to these images correctly.

### 7. Initialize Qdrant Collections

Ensure that the Qdrant collections specified in your `.env` file (`QDRANT_WARDROBE_COLLECTION` and `QDRANT_MARKETPLACE_COLLECTION`) exist in your Qdrant instance and are configured with the correct vector size for your CLIP model embeddings. You might need to create these collections manually or adapt the script if it includes collection creation logic (currently, it assumes they exist).

## Running the Application

Once the setup is complete, you can run the Streamlit application:

```bash
streamlit run app.py
```

This will start the web server, and you can access the application in your web browser, typically at `http://localhost:8501`.

## Technologies Used

- **Python:** Core programming language.
- **Streamlit:** For creating the web application interface.
- **PIL (Pillow):** For image manipulation.
- **Transformers (Hugging Face):** For using the CLIP model for image and text embeddings.
- **Qdrant Client:** For interacting with the Qdrant vector database (storing and searching image embeddings).
- **torch (PyTorch):** As a backend for the Transformers library.
- **SQLite3:** For the local database to save generated outfits.
- **dotenv:** For managing environment variables.
- **NumPy:** For numerical operations, especially with embeddings.

## How It Works

1.  **Image/Text Embedding:** When a user uploads a clothing item or enters a text prompt, the [`embeddings.py`](embeddings.py) script uses a pre-trained CLIP model (loaded in [`app.py`](app.py)) to generate numerical vector representations (embeddings) of the image or text.
2.  **Vector Database (Qdrant):** These embeddings are stored in a Qdrant vector database, managed by [`vector_db.py`](vector_db.py). Qdrant allows for efficient similarity searches.
    - Wardrobe items are stored in one collection.
    - Marketplace items are stored in another collection.
3.  **Outfit Generation:**
    - When a user requests an outfit based on a prompt, the text prompt is embedded.
    - The [`VectorDatabase.get_outfit_recommendations()`](vector_db.py) method in [`vector_db.py`](vector_db.py) queries Qdrant to find tops and bottoms from the user's wardrobe whose embeddings are similar to the prompt's embedding.
    - It then scores combinations of these tops and bottoms based on their similarity to the prompt and their coherence with each other (cosine similarity between their embeddings).
    - The application can also find items in the marketplace that are similar to an item from the user's wardrobe using [`VectorDatabase.get_similar_items_in_collection()`](vector_db.py).
4.  **Local Database (SQLite):**
    - The [`database.py`](database.py) script manages an SQLite database (`fashion.db`) to store user-saved outfits.
    - Functions like [`save_outfit()`](database.py) and [`get_saved_outfits()`](database.py) handle these operations.
5.  **Streamlit UI:**
    - [`app.py`](app.py) uses Streamlit to create the user interface, allowing users to navigate between pages, upload images, enter text, and view results.

## Future Improvements

- More sophisticated outfit scoring and recommendation algorithms.
- User accounts and personalized experiences.
- Direct integration with e-commerce platforms for marketplace items.
- Advanced filtering options in the wardrobe and marketplace.
- Ability to edit or delete items from the virtual wardrobe.

```# AI Fashion Assistant

The AI Fashion Assistant is a Streamlit web application that helps users manage their virtual wardrobe, generate outfit recommendations using AI, save their favorite outfits, and browse a thrift marketplace for new clothing items.

## Features

-   **Virtual Wardrobe:** Upload images of your clothing items, categorize them (top, bottom), and add descriptions.
-   **Outfit Generator:** Get AI-powered outfit recommendations based on a text prompt (e.g., "casual summer outfit"). It can also suggest outfits combining your wardrobe items with items from the marketplace.
-   **Saved Outfits:** Save your favorite generated outfits for future reference.
-   **Thrift Marketplace:** Browse and view details of clothing items available in a simulated thrift marketplace. Get outfit suggestions by combining marketplace items with your existing wardrobe.

## Project Structure

```

.
├── .env # Local environment variables (API keys, DB config)
├── .env.example # Example environment variables file
├── .gitignore # Specifies intentionally untracked files that Git should ignore
├── app.py # Main Streamlit application file
├── database.py # SQLite database interactions (saving/loading outfits)
├── embeddings.py # Functions for generating image and text embeddings using CLIP
├── fashion.db # SQLite database file
├── requirements.txt # Python dependencies
├── vector_db.py # Qdrant vector database interactions
├── clothes-images/ # Directory for user-uploaded wardrobe images (example)
├── marketplace-images/ # Directory for marketplace item images (example)
└── model/ # Directory for the pre-trained CLIP model

````

## Setup

### 1. Clone the Repository (if applicable)

```bash
git clone <your-repository-url>
cd <your-repository-directory>
````

### 2. Create a Virtual Environment

It's recommended to use a virtual environment to manage project dependencies.

```bash
python -m venv venv
```

Activate the virtual environment:

- On Windows:
  ```bash
  .\venv\Scripts\activate
  ```
- On macOS/Linux:
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies

Install the required Python packages using the [requirements.txt](requirements.txt) file:

```bash
pip install -r requirements.txt
```

### 4. Set up Environment Variables

Create a `.env` file in the root of the project directory by copying the [`.env.example`](.env.example) file:

```bash
cp .env.example .env
```

Edit the `.env` file and add your Qdrant API key and host URL:

```env
QDRANT_THRIFT_API_KEY='your_qdrant_api_key'
QDRANT_HOST='your_qdrant_endpoint_url'
QDRANT_WARDROBE_COLLECTION='clothes'
QDRANT_MARKETPLACE_COLLECTION='marketplace'
CLOTHING_TAGS = ["t-shirt", "shirt", ...] # Keep or modify as needed
```

### 5. Download the CLIP Model

The application uses a pre-trained CLIP model. You will need to download the model files and place them in a `model/` directory in the root of your project. The [`app.py`](app.py) file expects the model to be at `./model`.

You can typically download models from the Hugging Face Model Hub. For example, if you are using `openai/clip-vit-base-patch32`, ensure the `model/` directory contains files like `config.json`, `pytorch_model.bin`, `preprocessor_config.json`, etc.

### 6. Prepare Image Directories

- Create a directory named `images` (or `clothes-images` as per your `.gitignore`) in the root of your project. This is where images uploaded to the virtual wardrobe will be saved by default (see `show_wardrobe_page` function in [`app.py`](app.py)).
- If you have pre-existing marketplace images, ensure they are in a directory (e.g., `marketplace-images/`) and the paths in your Qdrant marketplace collection point to these images correctly.

### 7. Initialize Qdrant Collections

Ensure that the Qdrant collections specified in your `.env` file (`QDRANT_WARDROBE_COLLECTION` and `QDRANT_MARKETPLACE_COLLECTION`) exist in your Qdrant instance and are configured with the correct vector size for your CLIP model embeddings. You might need to create these collections manually or adapt the script if it includes collection creation logic (currently, it assumes they exist).

## Running the Application

Once the setup is complete, you can run the Streamlit application:

```bash
streamlit run app.py
```

This will start the web server, and you can access the application in your web browser, typically at `http://localhost:8501`.

## Technologies Used

- **Python:** Core programming language.
- **Streamlit:** For creating the web application interface.
- **PIL (Pillow):** For image manipulation.
- **Transformers (Hugging Face):** For using the CLIP model for image and text embeddings.
- **Qdrant Client:** For interacting with the Qdrant vector database (storing and searching image embeddings).
- **torch (PyTorch):** As a backend for the Transformers library.
- **SQLite3:** For the local database to save generated outfits.
- **dotenv:** For managing environment variables.
- **NumPy:** For numerical operations, especially with embeddings.

## How It Works

1.  **Image/Text Embedding:** When a user uploads a clothing item or enters a text prompt, the [`embeddings.py`](embeddings.py) script uses a pre-trained CLIP model (loaded in [`app.py`](app.py)) to generate numerical vector representations (embeddings) of the image or text.
2.  **Vector Database (Qdrant):** These embeddings are stored in a Qdrant vector database, managed by [`vector_db.py`](vector_db.py). Qdrant allows for efficient similarity searches.
    - Wardrobe items are stored in one collection.
    - Marketplace items are stored in another collection.
3.  **Outfit Generation:**
    - When a user requests an outfit based on a prompt, the text prompt is embedded.
    - The [`VectorDatabase.get_outfit_recommendations()`](vector_db.py) method in [`vector_db.py`](vector_db.py) queries Qdrant to find tops and bottoms from the user's wardrobe whose embeddings are similar to the prompt's embedding.
    - It then scores combinations of these tops and bottoms based on their similarity to the prompt and their coherence with each other (cosine similarity between their embeddings).
    - The application can also find items in the marketplace that are similar to an item from the user's wardrobe using [`VectorDatabase.get_similar_items_in_collection()`](vector_db.py).
4.  **Local Database (SQLite):**
    - The [`database.py`](database.py) script manages an SQLite database (`fashion.db`) to store user-saved outfits.
    - Functions like [`save_outfit()`](database.py) and [`get_saved_outfits()`](database.py) handle these operations.
5.  **Streamlit UI:**
    - [`app.py`](app.py) uses Streamlit to create the user interface, allowing users to navigate between pages, upload images, enter text, and view results.

## Future Improvements

- More sophisticated outfit scoring and recommendation algorithms.
- User accounts and personalized experiences.
- Direct integration with e-commerce platforms for marketplace items.
- Advanced filtering options in the wardrobe and marketplace.
- Ability to edit or delete items from the virtual wardrobe.
