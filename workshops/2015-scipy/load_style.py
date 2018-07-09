from IPython.display import display, HTML
import requests

def load_style(s):
    """Load a CSS stylesheet in the notebook by URL or filename.

    Examples::
    
        %load_style mystyle.css
        %load_style http://ipynbstyles.com/otherstyle.css
    """
    if s.startswith('http'):
        r =requests.get(s)
        style = r.text
    else:
        with open(s, 'r') as f:
            style = f.read()
    s = '<style>\n{style}\n</style>'.format(style=style)
    display(HTML(s))

def load_ipython_extension(ip):
    """Load the extension in IPython."""
    ip.register_magic_function(load_style)
