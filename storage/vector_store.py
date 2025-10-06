import chromadb

class VectorStore:
    def __init__(self):
        """initialize ChromaDB client and collection"""
        self.client = chromadb.Client()
        # if the collection does not exist, create one
        self.collection = self.client.get_or_create_collection("semantic_cache")

    def insert(self, embedding, query, response):
        """insert the vector, original question, and answer into the database"""
        # give each data a unique id
        item_id = str(len(self.collection.get()['ids']) + 1)
        self.collection.add(
            ids=[item_id],
            embeddings=[embedding],
            documents=[query],
            metadatas=[{"response": response}]
        )

    # keep the original search(k=1), return single (compatible with old code)
    def search(self, embedding, k=1):
        """search the most similar vector (default return the most similar one)"""
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=k
        )
        if results["documents"]:
            return {
                "query": results["documents"][0][0],
                "response": results["metadatas"][0][0]["response"],
                "distance": results["distances"][0][0]
            }
        return None

    # new: return the list of top-k
    def search_topk(self, embedding, k=3):
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=k
        )
        if not results["documents"]:
            return []

        items = []
        # the 0th dimension is "the nth query", here only 1 query, so fixed to [0]
        for i in range(len(results["documents"][0])):
            items.append({
                "query": results["documents"][0][i],
                "response": results["metadatas"][0][i]["response"],
                "distance": results["distances"][0][i]
            })
        # already sorted by distance from small to large, generally no need to sort again; for safety, sort again
        items.sort(key=lambda x: x["distance"])
        return items
