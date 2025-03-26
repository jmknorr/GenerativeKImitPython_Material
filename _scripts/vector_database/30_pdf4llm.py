#%% package
import pymupdf4llm
from pprint import pprint
#%% 
file_path = "Kursplan-Entwurf4-Feinentwurf-2025-02-23.pdf"
md_text = pymupdf4llm.to_markdown(file_path)
# %%
pprint(md_text)
#%% write to file
with open("Kursplan-Entwurf4-Feinentwurf-2025-02-23.md", "w") as f:
    f.write(md_text)

# %% extract images
md_text_images = pymupdf4llm.to_markdown(
    doc=file_path,
    pages=[20],
    page_chunks=True,
    write_images=True,
    image_path="images",
    image_format="png",
    dpi=300
)
md_text_images

# %% extract tables
md_text_tables = pymupdf4llm.to_markdown(
    doc=file_path,
    pages=[11] 
)

pprint(md_text_tables)
# %%
pprint(md_text_tables)
# %%
