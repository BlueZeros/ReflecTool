import os

class ProxyContext:
    def __init__(self, proxy):
        self.proxy = proxy
        self.original_http_proxy = os.environ.get('http_proxy')
        self.original_https_proxy = os.environ.get('https_proxy')

    def __enter__(self):
        os.environ['http_proxy'] = self.proxy
        os.environ['https_proxy'] = self.proxy

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_http_proxy is None:
            del os.environ['http_proxy']
        else:
            os.environ['http_proxy'] = self.original_http_proxy

        if self.original_https_proxy is None:
            del os.environ['https_proxy']
        else:
            os.environ['https_proxy'] = self.original_https_proxy