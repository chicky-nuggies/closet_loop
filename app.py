import streamlit as st
import os
from PIL import Image
from dotenv import load_dotenv
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import numpy as np
from embeddings import embed_image, embed_text
from vector_db import VectorDatabase
import torch
from database import init_db, save_outfit, get_saved_outfits

# Jank way to suppress error
torch.classes.__path__ = []

# Load environment variables
load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_API_KEY =os.getenv("QDRANT_THRIFT_API_KEY")
QDRANT_WARDROBE_COLLECTION = os.getenv('QDRANT_WARDROBE_COLLECTION')
QDRANT_MARKETPLACE_COLLECTION = os.getenv('QDRANT_MARKETPLACE_COLLECTION')
CLOTHING_TAGS = os.getenv('CLOTHING_TAGS')

# Initialize Qdrant client
client = QdrantClient(
    url=QDRANT_HOST,
    api_key=QDRANT_API_KEY,
)

db = VectorDatabase(QDRANT_HOST, QDRANT_API_KEY, QDRANT_WARDROBE_COLLECTION)
marketplace_db = VectorDatabase(QDRANT_HOST, QDRANT_API_KEY, QDRANT_MARKETPLACE_COLLECTION)

# Initialize CLIP model
@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    try:
        processor = AutoProcessor.from_pretrained("./model")
        model = AutoModelForZeroShotImageClassification.from_pretrained("./model")
        return processor, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

if 'model' not in st.session_state:
    st.session_state.processor, st.session_state.model = load_model()

# Initialize session state for generated outfits
if 'generated_outfits_data' not in st.session_state:
    st.session_state.generated_outfits_data = None

# Initialize the database when the app starts
init_db()

def main():
    st.title("AI Fashion Assistant 👔👗")
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Choose a page", 
        ["👔 Virtual Wardrobe", "🎨 Outfit Generator", "💾 Saved Outfits", "🛍️ Thrift Marketplace"]
    )
    
    if page == "👔 Virtual Wardrobe":
        show_wardrobe_page()
    elif page == "🎨 Outfit Generator":
        show_outfit_generator()
    elif page == "💾 Saved Outfits":
        show_saved_outfits()
    elif page == "🛍️ Thrift Marketplace":
        show_marketplace()

def show_wardrobe_page():
    st.header("My Virtual Wardrobe 👔")
    
    # Upload new item
    uploaded_file = st.file_uploader("Upload a new clothing item", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        # Show preview
        image = Image.open(uploaded_file)
        st.image(image, caption="Preview", use_column_width=True)
        
        # Get item details
        category = st.selectbox("Category", ["top", "bottom"])
        description = st.text_input("Description (optional)")
        
        if st.button("Add to Wardrobe"):
            try:
                # Save image
                save_path = os.path.join("images", uploaded_file.name)
                image.save(save_path)
                  # Get embedding
                embedding = embed_image(save_path, st.session_state.processor, st.session_state.model)
                
                # Add to Qdrant
                client.upsert(
                    collection_name=QDRANT_WARDROBE_COLLECTION,
                    points=[
                        PointStruct(
                            vector=embedding.tolist(),
                            payload={
                                "image_path": save_path,
                                "category": category,
                                "description": description
                            }
                        )
                    ]
                )
                st.success("Item added to wardrobe!")
            except Exception as e:
                st.error(f"Error adding item: {str(e)}")
    
    # Display wardrobe
    st.subheader("My Items")
    try:
        # Get all items
        scroll_result = client.scroll(
            collection_name=QDRANT_WARDROBE_COLLECTION,
            limit=100,
            with_payload=True
        )
        items = scroll_result[0]
        # wardrobe_items = db.get_all_items(limit=50) # You can adjust the limit as needed

        # st.write(items)
        # Display in grid
        cols = st.columns(3)
        for idx, item in enumerate(items):
            with cols[idx % 3]:
                if os.path.exists(item.payload["image_path"]):
                    st.image(item.payload["image_path"])
                    st.caption(f"{item.payload.get('product_name', 'No description')}")
    except Exception as e:
        st.error(f"Error loading wardrobe: {str(e)}")

def show_outfit_generator():
    st.header("AI Outfit Generator 🎨")
    
    prompt_input = st.text_input("Describe the outfit you want", "casual summer outfit")

    if st.button("Generate Outfits"):
        try:
            with st.spinner("Generating outfit recommendations..."):
                query_embedding = embed_text(prompt_input, st.session_state.processor, st.session_state.model)
                outfits = db.get_outfit_recommendations(query_embedding, limit=3)
                
                marketplace_bottom_hits_results = [] # Renamed from potential_outfits_results
                base_top_for_potential_outfits = None

                if outfits and outfits[0].get('bottom') and outfits[0]['bottom'].get('id'):
                    base_top_for_potential_outfits = outfits[0]['bottom']
                    # Use marketplace_db for marketplace items
                    marketplace_bottom_hits_results = marketplace_db.get_similar_items_in_collection( # Renamed
                        base_top_for_potential_outfits['id'], 
                        'clothes', 
                        QDRANT_MARKETPLACE_COLLECTION, # Ensure this is the correct marketplace collection name
                        'top' 
                    )
                
                if not outfits:
                    st.warning("Not enough items in your wardrobe or no matching outfits found. Please add more clothes!")
                    st.session_state.generated_outfits_data = {
                        "prompt": prompt_input, 
                        "outfits": [],
                        "marketplace_bottom_hits": [], # Renamed key
                        "base_top_for_potential": None
                    }
                else:
                    st.session_state.generated_outfits_data = {
                        "prompt": prompt_input, 
                        "outfits": outfits,
                        "marketplace_bottom_hits": marketplace_bottom_hits_results, # Renamed key and assigned renamed variable
                        "base_top_for_potential": base_top_for_potential_outfits
                    }
                    st.success("✨ Generated outfit recommendations based on your style!")
        except Exception as e:
            st.error(f"Error generating outfits: {str(e)}")
            st.session_state.generated_outfits_data = None 

    if st.session_state.generated_outfits_data:
        current_prompt = st.session_state.generated_outfits_data["prompt"]
        outfits_to_display = st.session_state.generated_outfits_data["outfits"]
        # Renamed variable and updated key for .get()
        marketplace_bottoms_to_display = st.session_state.generated_outfits_data.get("marketplace_bottom_hits", []) 
        base_top_item = st.session_state.generated_outfits_data.get("base_top_for_potential")
        
        # st.write(marketplace_bottoms_to_display) # If you had a debug print, update variable name

        if outfits_to_display:
            st.markdown("---") 
            st.subheader("Recommended Outfits (From Your Wardrobe)")

            for idx, outfit in enumerate(outfits_to_display):
                col1, col2 = st.columns(2)
                with col1:
                    if "top" in outfit and outfit["top"] and "image_path" in outfit["top"] and os.path.exists(outfit["top"]["image_path"]):
                        st.image(outfit["top"]["image_path"], 
                               caption=outfit["top"].get("product_name", "Top"))
                    else:
                        st.write("Top image not available.")
                with col2:
                    if "bottom" in outfit and outfit["bottom"] and "image_path" in outfit["bottom"] and os.path.exists(outfit["bottom"]["image_path"]):
                        st.image(outfit["bottom"]["image_path"], 
                               caption=outfit["bottom"].get("product_name", "Bottom"))
                    else:
                        st.write("Bottom image not available.")
                
                # st.markdown(f"Match Score: {outfit.get('score', 0.0):.2f}")
                
                save_button_key = f"save_outfit_{idx}"
                if st.button("💾 Save Outfit", key=save_button_key):
                    try:
                        outfit_to_save = {
                            "top": outfit["top"],
                            "bottom": outfit["bottom"],
                            # "score": outfit["score"],
                            "prompt": current_prompt
                        }
                        save_outfit(outfit_to_save)
                        st.success(f"Outfit {idx + 1} saved!")
                    except Exception as e:
                        st.error(f"Failed to save outfit {idx + 1}: {str(e)}")
                st.markdown("---")
        elif not outfits_to_display and current_prompt: 
            pass

        # Display Potential Outfits from Marketplace
        # Updated condition to use renamed variable
        if marketplace_bottoms_to_display and base_top_item: 
            st.subheader("Potential Outfits (Your Pieces + Marketplace Pieces)")
            st.write(f"Your piece: {base_top_item.get('description', 'Top')}")
            if "image_path" in base_top_item and os.path.exists(base_top_item["image_path"]): # Added check for image_path key
                st.image(base_top_item["image_path"], width=150) 
            else:
                st.write("Base top image not available.")
            st.markdown("---")

            # Updated loop to use renamed variable
            for idx, potential_item_hit in enumerate(marketplace_bottoms_to_display): 
                potential_item = potential_item_hit 
                col1, col2 = st.columns(2)
                
                with col1: 
                    st.write("Marketplace Bottom:")
                    if "image_path" in potential_item.payload and os.path.exists(potential_item.payload['image_path']):
                        st.image(potential_item.payload["image_path"], caption=potential_item.payload['product_name'])
                    else:
                        st.write("Marketplace item image not available.")
                    st.write(f"Name: {potential_item.payload['product_name']}")
                    st.write(f"Price: RM {potential_item.payload['price']}")
                    st.write(f"Store: Beyond Encore")
                    # st.write(f"Similarity Score: {potential_item_hit.score:.2f}")

                
                with col2: 
                    st.write("Your Top:")
                    if "image_path" in base_top_item and os.path.exists(base_top_item["image_path"]): # Added check for image_path key
                        st.image(base_top_item["image_path"], caption=base_top_item.get("description", "Top"))
                    else:
                        st.write("Top image not available.")

                save_button_key = f"save_outfit_{idx+500}"
                if st.button("💾 Save Outfit", save_button_key):
                    try:
                        outfit_to_save = {
                            "top": base_top_item,
                            "bottom": potential_item.payload,
                            # "score": outfit["score"],
                            "prompt": current_prompt
                        }
                        save_outfit(outfit_to_save)
                        st.success(f"Outfit {idx + 1} saved!")
                    except Exception as e:
                        st.error(f"Failed to save outfit {idx + 1}: {str(e)}")


                
                st.markdown("---")
        # Updated condition to use renamed variable
        elif base_top_item and not marketplace_bottoms_to_display and outfits_to_display: 
            st.info("No matching bottoms found in the marketplace for your selected top to create potential outfits.")


def show_saved_outfits():
    st.header("Saved Outfits 💾")
    
    saved_outfits = get_saved_outfits()
    if not saved_outfits:
        st.info("No outfits saved yet.")
    else:
        # st.write(saved_outfits)
        for outfit in saved_outfits:
            st.markdown(f"### Outfit {outfit['id']}")
            col1, col2 = st.columns(2)
            with col1:
                st.image(outfit["top_image"], 
                        caption=outfit["top_description"] or "Top")
            with col2:
                st.image(outfit["bottom_image"], 
                        caption=outfit["bottom_description"] or "Bottom")
            st.markdown(f"Prompt: {outfit['prompt']}")
            # st.markdown(f"Match Score: {outfit['score']:.2f}")
            st.markdown(f"Saved on: {outfit['created_at']}")
            st.markdown("---")

def show_marketplace():
    st.header("Thrift Marketplace 🛍️")

    # Initialize session state for selected item
    if 'selected_marketplace_item' not in st.session_state:
        st.session_state.selected_marketplace_item = None

    marketplace_items = marketplace_db.get_all_items(limit=50) # You can adjust the limit as needed


    if st.session_state.selected_marketplace_item:
        # --- Detailed Item View ---
        item = st.session_state.selected_marketplace_item
        
        col1, col2 = st.columns([1, 2]) # Adjust column ratios as needed
        with col1:
            if os.path.exists(item.payload["image_path"]):
                st.image(item.payload["image_path"], use_container_width=True)
            else:
                st.warning("Image not found.")
        
        with col2:
            st.subheader(item.payload["product_name"])
            st.write(f"**Price:** {f"RM {item.payload['price']}"}")
            st.write(f"**Tags:** {item.payload["tags"]}")
            # You can add more details here, e.g., size options, material, seller info

        if st.button("⬅️ Back to Marketplace"):
            st.session_state.selected_marketplace_item = None
            st.rerun() # Use st.rerun() for Streamlit v1.28.0+

        # --- Generate and Display Outfits ---
        st.markdown("---")
        st.subheader("Outfit Ideas with this Item")

        item_payload = item.payload
        
        if "category" not in item_payload:
            st.warning("Cannot generate outfits: Marketplace item category is missing.")
        elif not item_payload["category"]:
            st.warning("Cannot generate outfits: Marketplace item category is not defined.")
        else:
            item_category = item_payload["category"]
            item_id = item.id 

            complementary_category = "bottom" if item_category.lower() == "top" else "top"

            try:
                # Use 'db' (wardrobe) to find items in the wardrobe collection
                # that complement the marketplace item.
                # The get_similar_items_in_collection method is called on the 'db' (wardrobe) instance.
                # It will use the item_id from the marketplace_db.collection_name 
                # to find its vector, and then search for complementary items 
                # in the db.collection_name (wardrobe).
                complementary_wardrobe_items = db.get_similar_items_in_collection(
                    item_id=item_id,
                    origin_collection_name=marketplace_db.collection_name, # Collection of the marketplace item
                    target_collection_name=db.collection_name,             # Wardrobe collection to search in
                    filter=complementary_category,
                    limit=2 # Generate 2 outfits
                )

                if not complementary_wardrobe_items:
                    st.info(f"Could not find any matching '{complementary_category}' items in your wardrobe to pair with this marketplace item.")
                else:
                    st.write(f"Here are some outfit ideas combining this marketplace {item_category.lower()} with items from your wardrobe:")
                    for wardrobe_item_hit in complementary_wardrobe_items:
                        wardrobe_item_payload = wardrobe_item_hit.payload
                        st.markdown("---")
                        cols_outfit = st.columns(2)

                        # Display based on which is top/bottom
                        if item_category.lower() == "top":
                            # Marketplace item is TOP, Wardrobe item is BOTTOM
                            with cols_outfit[0]:
                                st.write(f"**Marketplace Item (Top):** {item_payload.get('product_name', 'Top')}")
                                if os.path.exists(item_payload["image_path"]):
                                    st.image(item_payload["image_path"], use_container_width=True)
                                else:
                                    st.caption("Image not available")

                            with cols_outfit[1]:
                                st.write(f"**Your Wardrobe (Bottom):** {wardrobe_item_payload.get('product_name', 'Bottom')}")
                                if os.path.exists(wardrobe_item_payload["image_path"]):
                                    st.image(wardrobe_item_payload["image_path"], use_container_width=True)
                                else:
                                    st.caption("Image not available")
                        else: # Marketplace item is BOTTOM, Wardrobe item is TOP
                            with cols_outfit[0]:
                                st.write(f"**Your Wardrobe (Top):** {wardrobe_item_payload.get('product_name', 'Top')}")
                                if os.path.exists(wardrobe_item_payload["image_path"]):
                                    st.image(wardrobe_item_payload["image_path"], use_container_width=True)
                                else:
                                    st.caption("Image not available")

                            with cols_outfit[1]:
                                st.write(f"**Marketplace Item (Bottom):** {item_payload.get('product_name', 'Bottom')}")
                                if os.path.exists(item_payload["image_path"]):
                                    st.image(item_payload["image_path"], use_container_width=True)
                                else:
                                    st.caption("Image not available")
            except Exception as e:
                st.error(f"Error generating outfit suggestions: {str(e)}")
                st.error("Please ensure the marketplace item has a 'category' (e.g., 'top', 'bottom') in its data and that wardrobe items are available.")

    else:
        # --- Grid View of Marketplace Items ---
        st.subheader("Browse Items")
        # st.write(marketplace_items)
        num_items = len(marketplace_items)
        cols_per_row = 3 # Number of columns for the grid

        for i in range(0, num_items, cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if i + j < num_items:
                    item = marketplace_items[i+j]
                    with cols[j]:
                        container = st.container()
                        if os.path.exists(item.payload["image_path"]):
                            container.image(item.payload["image_path"], use_container_width='auto')
                        else:
                            container.caption("Image not available")
                        container.write(item.payload["product_name"])
                        container.write(f"RM {item.payload['price']}")
                        if container.button("View Details", key=f"details_{item.id}"):
                            st.session_state.selected_marketplace_item = item
                            st.rerun() # Use st.rerun() for Streamlit v1.28.0+

if __name__ == "__main__":
    main()
