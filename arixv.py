from typing import List, Optional
from langchain_community.retrievers import ArxivRetriever
from langchain_core.documents import Document

class EnhancedArxivRetriever(ArxivRetriever):
    """
    An enhanced version of ArxivRetriever that extracts richer metadata 
    such as primary categories, journal references, and DOIs.
    """
    
    include_extra_metadata: bool = True

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        # Fetch documents using the standard parent class logic
        docs = super()._get_relevant_documents(query, run_manager=run_manager)
        
        if not self.include_extra_metadata:
            return docs

        try:
            import arxiv
        except ImportError:
            return docs

        # Re-query or map using the arxiv client to fetch extra fields 
        # depending on how deep you want to integrate it into the wrapper.
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=len(docs))
        
        enhanced_docs = []
        for doc, result in zip(docs, client.results(search)):
            # Inject additional metadata fields
            doc.metadata.update({
                "Primary Category": getattr(result, "primary_category", None),
                "Journal Reference": getattr(result, "journal_ref", None),
                "DOI": getattr(result, "doi", None),
                "PDF URL": getattr(result, "pdf_url", None),
            })
            enhanced_docs.append(doc)
            
        return enhanced_docs

# --- Example Usage ---
if __name__ == "__main__":
    retriever = EnhancedArxivRetriever(
        load_max_docs=2,
        load_all_available_meta=True
    )

    docs = retriever.invoke("large language models")

    for i, doc in enumerate(docs):
        print(f"\n--- Result {i+1} ---")
        print("Title:", doc.metadata.get("Title"))
        print("Authors:", doc.metadata.get("Authors"))
        print("Primary Category:", doc.metadata.get("Primary Category"))
        print("DOI:", doc.metadata.get("DOI"))
        print("PDF URL:", doc.metadata.get("PDF URL"))
        print("Summary:", doc.page_content[:300] + "...")
