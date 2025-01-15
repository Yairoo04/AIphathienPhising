import re
from urllib.parse import urlparse

def extract_features(url):
    try:
        # Thêm scheme mặc định nếu thiếu
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url

        # Phân tích URL
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        if not domain:  # Kiểm tra tính hợp lệ của domain
            raise ValueError(f"Invalid URL: {url}")

        # Trích xuất các đặc trưng từ URL
        return {
            'url_length': len(url),
            'num_special_chars': len(re.findall(r'[?|#|=|&]', url)),
            'is_https': 1 if parsed_url.scheme == "https" else 0,
            'num_digits': len(re.findall(r'\d', url)),
            'domain_length': len(domain),
            'num_subdomains': max(len(domain.split('.')) - 2, 0),  # Tránh số âm
            'num_dashes': domain.count('-'),
            'path_length': len(parsed_url.path),
            'query_length': len(parsed_url.query),
            'has_ip': 1 if re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', domain) else 0,  # Kiểm tra IP
            'has_at_symbol': 1 if '@' in url else 0,  # Kiểm tra ký tự "@"
            'redirect_count': url.count('http') - 1  # Đếm số lần chuyển hướng (http xuất hiện)
        }
    except Exception as e:
        # Ghi log lỗi thay vì in ra màn hình
        print(f"Error extracting features from URL '{url}': {e}")
        return {
            'url_length': -1,           # Giá trị mặc định
            'num_special_chars': -1,   # Giá trị mặc định
            'is_https': -1,            # Giá trị mặc định
            'num_digits': -1,          # Giá trị mặc định
            'domain_length': -1,       # Giá trị mặc định
            'num_subdomains': -1,      # Giá trị mặc định
            'num_dashes': -1,          # Giá trị mặc định
            'path_length': -1,         # Giá trị mặc định
            'query_length': -1,        # Giá trị mặc định
            'has_ip': -1,              # Giá trị mặc định
            'has_at_symbol': -1,       # Giá trị mặc định
            'redirect_count': -1       # Giá trị mặc định
        }
