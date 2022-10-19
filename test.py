import gutenbergpy.textget
#this gets a book by its gutenberg id
raw_book    = gutenbergpy.textget.get_text_by_id(1000)
print(raw_book)
#this strips the headers from the book
clean_book  = gutenbergpy.textget.strip_headers(raw_book)
print(clean_book)