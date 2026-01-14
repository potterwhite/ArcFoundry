# Copyright (c) 2026 PotterWhite
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import requests
from tqdm import tqdm
from core.utils import logger, ensure_dir

class ModelDownloader:
    """
    Manages model assets: checks local existence or downloads from URL.
    """
    
    def ensure_model(self, local_path, url=None):
        """
        Ensures the model exists at local_path.
        1. Check if exists locally -> Return True
        2. If not and url provided -> Download -> Return True
        3. If not and no url -> Return False
        """
        if os.path.exists(local_path):
            logger.info(f"✔ [Offline] Found local model: {local_path}")
            return True
            
        if not url:
            logger.error(f"❌ [Missing] Model not found locally and no URL provided: {local_path}")
            return False
            
        # Start Download
        logger.info(f"⬇ [Download] Model missing. Downloading from: {url}")
        return self._download_file(url, local_path)

    def _download_file(self, url, dest_path):
        try:
            ensure_dir(os.path.dirname(dest_path))
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 # 1KB
            
            with open(dest_path, 'wb') as file, tqdm(
                desc=os.path.basename(dest_path),
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(block_size):
                    size = file.write(data)
                    bar.update(size)
            
            logger.info(f"✔ Download complete: {dest_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            # Clean up partial file
            if os.path.exists(dest_path):
                os.remove(dest_path)
            return False
