import hashlib
import chromadb
import pymupdf

DATA_COLLECTION_NAME = "course_materials"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(name=DATA_COLLECTION_NAME)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """this func cuts the large text into overlapping pieces"""

    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size-overlap
    return chunks


def add_pdf(pdf_bytes: bytes, source_name: str) -> int:
    """split each page on the pdf into chunks and return the number of chunks """
    
    file_id = hashlib.md5(pdf_bytes).hexdigest()
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")

    ids = []
    texts = []
    metadata = []

    for page_number, page in enumerate(doc, start=1):
        page_text = page.get_text().strip()
        if not page_text:
            continue

        page_chunks = chunk_text(page_text)

        for i, chunk in enumerate(page_chunks):
            ids.append(f"{source_name}-p{page_number}-c{i}")
            texts.append(chunk)
            metadata.append({"source": source_name, "page": page_number})

    doc.close()

    if texts:
        collection.upsert(ids=ids, documents=texts, metadatas=metadata)

    return len(texts)


def search(query: str, num_of_results: int = 4) -> list[dict]:
    """find the chunks that is closest in meaning to the query"""

    results = collection.query(query_texts=[query], n_results=num_of_results)

    found = []

    for text, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        found.append({
            "text": text,
            "source": meta["source"],
            "page": meta["page"],
            "distance": distance
        })
    return found


if __name__ == "__main__":
    with open("data/reference.pdf", "rb") as f:
        count = add_pdf(f.read(), "reference.pdf")
    print(f"Stored {count} chunks.")

    for hit in search("Algorithmics is more than a branch of computer science. It is the core of computer science, and, in all "):
        print(f"\n[{hit['source']} p.{hit['page']}] distance={hit['distance']:.3f}")
        print(hit["text"][:300])
