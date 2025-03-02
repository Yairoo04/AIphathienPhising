import re
from urllib.parse import urlparse

def extract_features(url):
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url

        # Phân tích URL
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        if not domain:
            raise ValueError(f"Invalid URL: {url}")

        # Trích xuất các đặc trưng từ URL
        return {
            'url_length': len(url),
            'num_special_chars': len(re.findall(r'[?|#|=|&]', url)),
            'is_https': 1 if parsed_url.scheme == "https" else 0,
            'num_digits': len(re.findall(r'\d', url)),
            'domain_length': len(domain),
            'num_subdomains': max(len(domain.split('.')) - 2, 0),
            'num_dashes': domain.count('-'),
            'path_length': len(parsed_url.path),
            'query_length': len(parsed_url.query),
            'has_ip': 1 if re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', domain) else 0,
            'has_at_symbol': 1 if '@' in url else 0,
            'redirect_count': url.count('http') - 1 
        }
    except Exception as e:
        print(f"Error extracting features from URL '{url}': {e}")
        return {
            'url_length': -1,           
            'num_special_chars': -1,   
            'is_https': -1,            
            'num_digits': -1,          
            'domain_length': -1,       
            'num_subdomains': -1,      
            'num_dashes': -1,          
            'path_length': -1,         
            'query_length': -1,        
            'has_ip': -1,              
            'has_at_symbol': -1,       
            'redirect_count': -1       
        }
