import os
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct
from embeddings import embed_text, embed_image

class VectorDatabase:
    def __init__(self, host: str, api_key: str, collection_name: str):
        """Initialize the vector database connection"""
        self.client = QdrantClient(
            url=host,
            api_key=api_key,
        )
        self.collection_name = collection_name

    def add_item(self, image_path: str, category: str, description: str, embedding: np.ndarray, clothing_tags: list[str]):
        """Add a new clothing item to the database"""
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        vector=embedding.tolist(),
                        payload={
                            "image_path": image_path,
                            "category": category,
                            "description": description
                        }
                    )
                ]
            )
            return True
        except Exception as e:
            raise Exception(f"Error adding item: {str(e)}")

    def get_items_by_category(self, category: str, query_embedding: np.ndarray, limit: int = 5):
        """Get items by category with similarity search"""
        try:
            return self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding.tolist(),
                query_filter=Filter(
                    must=[FieldCondition(key="category", match=MatchValue(value=category))]
                ),
                with_vectors=True,
                with_payload=True,
                limit=limit
            ).points
        except Exception as e:
            raise Exception(f"Error querying items: {str(e)}")

    def get_all_items(self, limit: int = 100):
        """Get all items in the wardrobe"""
        try:
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                with_payload=True
            )
            return scroll_result[0]
        except Exception as e:
            raise Exception(f"Error loading wardrobe: {str(e)}")
        
    def get_item_by_id(self, item_id: str):
        """Get a specific item by its ID"""
        try:
            # The retrieve method returns a list of points, even if only one ID is requested.
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[item_id],
                with_payload=True,
                with_vectors=True  # Or False, depending on whether you need the vector
            )
            if points:
                return points[0]  # Return the first (and should be only) point
            return None
        except Exception as e:
            raise Exception(f"Error retrieving item with ID '{item_id}': {str(e)}")

        
    def get_similar_items_in_collection(self, item_id: int ,origin_collection_name:str,  target_collection_name: str, filter: str, limit: int = 2):
        """
        Get similar items from a specified target collection based on a query embedding.
        This can be used to find items in one collection (e.g., marketplace)
        that are coherent with an item from another collection (e.g., wardrobe).
        """
        item_vector = self.client.retrieve(
            collection_name=origin_collection_name,
            ids=[item_id],
            with_vectors=True
        )[0].vector

        try:
            search_results = self.client.query_points(
                collection_name=target_collection_name,
                query=item_vector,
                with_payload=True,
                query_filter=Filter(
                must=[FieldCondition(key="category", match=MatchValue(value=filter))]
                ),
                limit=limit
            )
            # Returns one image path
            return search_results.points
        except Exception as e:
            raise Exception(f"Error querying similar items from collection '{target_collection_name}': {str(e)}")


    def get_outfit_recommendations(self, query_embedding: np.ndarray, limit: int = 5):
        """Get outfit recommendations based on a query"""
        try:
            # Get top candidates
            tops = self.get_items_by_category("top", query_embedding, limit)
            # Get bottom candidates
            bottoms = self.get_items_by_category("bottom", query_embedding, limit)

            if not tops or not bottoms:
                return []

            # Score and get outfit pairs
            return self._score_outfit_combinations(query_embedding, tops, bottoms, limit)
        except Exception as e:
            raise Exception(f"Error generating recommendations: {str(e)}")

    def _score_outfit_combinations(self, query_embedding, tops, bottoms, limit: int = 3):
        """Score outfit combinations based on coherence and query relevance"""
        pair_scores = []
        for top in tops:
            for bottom in bottoms:
                top_vector = np.array(top.vector)
                bottom_vector = np.array(bottom.vector)
                
                coherence = self._cosine_similarity(top_vector, bottom_vector)
                query_relevance = (
                    self._cosine_similarity(query_embedding, top_vector) +
                    self._cosine_similarity(query_embedding, bottom_vector)
                ) / 2
                top.payload['id'] = top.id
                bottom.payload['id'] = bottom.id
                score = 0.5 * query_relevance + 0.5 * coherence
                pair_scores.append({
                    "score": score,
                    "top": top.payload,
                    "bottom": bottom.payload,
                })

        pair_scores.sort(key=lambda x: x["score"], reverse=True)
        return pair_scores[:limit]

    @staticmethod
    def _cosine_similarity(vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))