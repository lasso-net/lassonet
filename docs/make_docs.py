import os
import re

import markdown2

os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open("template.html") as f:
    template = f.read()


def replace(match):
    filename = match.group(1) + ".md"
    return markdown2.markdown_path(filename)


index = re.sub(r"{{(\w+)}}", replace, template)

with open("index.html", "w") as f:
    f.write(index)
