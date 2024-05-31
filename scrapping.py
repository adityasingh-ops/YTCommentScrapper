import googleapiclient.discovery
import re
from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from sklearn.cluster import KMeans
import numpy as np

# Step 1: Scrape YouTube Comments
def get_youtube_comments(video_id, api_key):
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)
    
    comments = []
    request = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=100)
    
    while request:
        response = request.execute()
        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textOriginal"]
            comments.append(comment)
        
        request = youtube.commentThreads().list_next(request, response)
    
    return comments

def save_comments_to_file(comments, filename):
    with open(filename, 'w') as file:
        for comment in comments:
            file.write(comment + '\n')

# Step 2: Preprocess Comments
def preprocess_comments(comments):
    cleaned_comments = []
    for comment in comments:
        comment = re.sub(r'\W+', ' ', comment)  # Remove special characters
        comment = comment.lower()  # Convert to lowercase
        cleaned_comments.append(comment)
    return cleaned_comments

# Step 3: Convert Comments to Vectors
def embed_comments(comments):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(comments)
    return embeddings

# Step 4: Store Vectors in a Vector Database (Milvus)
def store_vectors_in_milvus(embeddings):
    connections.connect("default", host="localhost", port="19530")
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embeddings.shape[1])
    ]
    schema = CollectionSchema(fields, "Comment embeddings")
    collection = Collection("youtube_comments", schema)
    data = [
        [i for i in range(len(embeddings))],
        embeddings.tolist()
    ]
    collection.insert(data)
    index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}
    collection.create_index("embedding", index_params)
    collection.load()
    return collection

# Step 5: Cluster Comments by Similarity
def cluster_comments(embeddings, num_clusters=5):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(embeddings)
    labels = kmeans.labels_
    return labels

# Step 6: Display Results
def display_clustered_comments(comments, labels):
    grouped_comments = {}
    for label, comment in zip(labels, comments):
        if label not in grouped_comments:
            grouped_comments[label] = []
        grouped_comments[label].append(comment)
    for cluster, grouped in grouped_comments.items():
        print(f"Cluster {cluster}:")
        for comment in grouped:
            print(f"- {comment}")
        print("\n")

# Main script execution
if __name__ == "__main__":
    api_key = "AIzaSyDvknRGIztn64cmL5MJDmyJP0ySpJ3bipo"  # Replace with your actual API key
    video_id = "ODA3rWfmzg8"  # Replace with the actual video ID

    # Step 1: Scrape YouTube Comments
    comments = get_youtube_comments(video_id, api_key)
    

    # # Step 2: Preprocess Comments
    cleaned_comments = preprocess_comments(comments)
    save_comments_to_file(cleaned_comments, "youtube_comments.txt")

    # Step 3: Convert Comments to Vectors
    embeddings = embed_comments(cleaned_comments)
    np.savetxt("comment_embeddings.txt", embeddings, delimiter=",")

    # Step 4: Store Vectors in a Vector Database (Milvus)
    collection = store_vectors_in_milvus(embeddings)

    # Step 5: Cluster Comments by Similarity
    labels = cluster_comments(embeddings)

    # Step 6: Display Results
    display_clustered_comments(comments, labels)
