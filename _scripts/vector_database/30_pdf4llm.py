
#%% package
import pymupdf4llm
from pprint import pprint
#%% 
file_path = "data/DE102013210737A1.pdf"
md_text = pymupdf4llm.to_markdown(file_path, pages=[1])
# %%
pprint(md_text)
#%% write to file

# %% extract images
md_text_images = pymupdf4llm.to_markdown(
    doc=file_path,
    pages=[13],
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
