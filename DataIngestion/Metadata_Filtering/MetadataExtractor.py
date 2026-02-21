import re
import fitz
from langchain_core.documents  import Document


class MetadataExtractor:
    
 

    def extract_content_with_subject_metadata(self, pdf_path):
        """
        Parse TEKS PDF and attach correct subject metadata
        to each content line dynamically.
        """

        doc = fitz.open(pdf_path)

        # Detect subject header like:
        # §111.7. Mathematics, Grade 5, Adopted 2012.
        subject_pattern = re.compile(
            r"§\d+\.\d+\.\s+(.+?),\s+Grade\s+\d+",
            re.IGNORECASE
        )

        documents = []
        current_subject = None

        for page_number, page in enumerate(doc):
            text = page.get_text()
            lines = text.split("\n")

            for line in lines:
                line = line.strip()

                # Skip empty lines or footer noise
                if not line or "revised August" in line:
                    continue

                # -----------------------------
                # Detect new subject header
                # -----------------------------
                subject_match = subject_pattern.search(line)
                if subject_match:
                    current_subject = subject_match.group(1).strip()
                    continue  # Skip storing the header line itself

                # -----------------------------
                # Store content with subject metadata
                # -----------------------------
                documents.append(
                    Document(
                        page_content=line,
                        metadata={
                            "subject": current_subject,
                            "page": page_number + 1
                        }
                    )
                )

        return documents

    