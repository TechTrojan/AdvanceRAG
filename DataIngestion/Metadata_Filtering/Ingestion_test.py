 

from MetadataExtractor import MetadataExtractor
import os 

me = MetadataExtractor()
cwd = os.getcwd()



file_path = f"{cwd}\DataIngestion\Metadata_Filtering\grade5-teks-062024-0.pdf"

#print(file_path)

docs = me.extract_content_with_subject_metadata(file_path)

print(docs[5:15])



