import os
import re
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RBIScraper:
    """Scraper for RBI regulatory documents.
    
    This class handles the scraping of circulars, notifications, and other
    regulatory documents from the Reserve Bank of India website.
    """
    
    BASE_URL = "https://www.rbi.org.in"
    NOTIFICATIONS_URL = f"{BASE_URL}/Scripts/NotificationUser.aspx"
    CIRCULARS_URL = f"{BASE_URL}/Scripts/BS_CircularIndexDisplay.aspx"
    
    def __init__(self, output_dir: str = "data/raw"):
        """Initialize the RBI scraper.
        
        Args:
            output_dir: Directory to store scraped documents
        """
        self.output_dir = Path(output_dir)
        self.session = requests.Session()
        
        # Set up headers to mimic a browser
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        self.session.headers.update(self.headers)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.output_dir / "circulars", exist_ok=True)
        os.makedirs(self.output_dir / "notifications", exist_ok=True)
    
    def _make_request(self, url: str, params: Optional[Dict] = None, delay: float = 1.0) -> requests.Response:
        """Make an HTTP request with rate limiting.
        
        Args:
            url: URL to request
            params: Optional query parameters
            delay: Time to wait between requests in seconds
            
        Returns:
            Response object
        """
        time.sleep(delay)  # Rate limiting
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise
    
    def _extract_document_metadata(self, soup: BeautifulSoup) -> Dict:
        """Extract metadata from a document page.
        
        Args:
            soup: BeautifulSoup object of the document page
            
        Returns:
            Dictionary containing document metadata
        """
        metadata = {}
        
        # Extract title
        title_elem = soup.find("span", {"id": "lblheading"}) or soup.find("h1") or soup.find("title")
        if title_elem:
            metadata["title"] = title_elem.text.strip()
        
        # Extract date
        date_pattern = re.compile(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b')
        date_matches = date_pattern.findall(str(soup))
        if date_matches:
            metadata["date"] = date_matches[0]
        
        # Extract document ID/reference
        ref_pattern = re.compile(r'\b(?:RBI|DOR|DBOD|DNBR)[\w.-]+/\d{2,4}(?:-\d{2,4})?/\d+\b')
        ref_matches = ref_pattern.findall(str(soup))
        if ref_matches:
            metadata["reference_id"] = ref_matches[0]
        
        # Extract addressee/category if available
        addressee_elem = soup.find("p", string=re.compile(r"All|To The|The Chairman"))
        if addressee_elem:
            metadata["addressee"] = addressee_elem.text.strip()
        
        return metadata
    
    def _save_document(self, content: str, metadata: Dict, doc_type: str, url: str) -> str:
        """Save document content and metadata to file.
        
        Args:
            content: HTML content of the document
            metadata: Document metadata
            doc_type: Type of document (circular, notification)
            url: Source URL
            
        Returns:
            Path to saved file
        """
        # Create a filename based on date and reference ID if available
        ref_id = metadata.get("reference_id", "")
        date_str = metadata.get("date", datetime.now().strftime("%Y-%m-%d"))
        try:
            date_obj = datetime.strptime(date_str, "%B %d, %Y")
            date_formatted = date_obj.strftime("%Y-%m-%d")
        except ValueError:
            date_formatted = date_str.replace(",", "").replace(" ", "_")
        
        filename = f"{date_formatted}_{ref_id}".replace("/", "_").replace("\\", "_")
        if not filename.strip("_"):
            # Fallback if we couldn't extract meaningful filename components
            filename = f"rbi_{doc_type}_{int(time.time())}"
        
        # Save HTML content
        file_path = self.output_dir / doc_type / f"{filename}.html"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        # Save metadata as JSON
        metadata_path = self.output_dir / doc_type / f"{filename}.metadata.txt"
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write(f"URL: {url}\n")
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        
        return str(file_path)
    
    def scrape_document(self, url: str, doc_type: str = "circulars") -> Tuple[str, Dict]:
        """Scrape a single document from its URL.
        
        Args:
            url: URL of the document
            doc_type: Type of document (circular, notification)
            
        Returns:
            Tuple of (file_path, metadata)
        """
        logger.info(f"Scraping document: {url}")
        
        # Make the request
        response = self._make_request(url)
        content = response.text
        soup = BeautifulSoup(content, "lxml")
        
        # Extract metadata
        metadata = self._extract_document_metadata(soup)
        
        # Check for PDF link
        pdf_link = None
        for link in soup.find_all("a", href=True):
            if link.get("href", "").lower().endswith(".pdf"):
                pdf_link = link["href"]
                if not pdf_link.startswith("http"):
                    pdf_link = self.BASE_URL + pdf_link
                metadata["pdf_url"] = pdf_link
                break
        
        # Save the document
        file_path = self._save_document(content, metadata, doc_type, url)
        
        # Download PDF if available
        if pdf_link:
            try:
                logger.info(f"Downloading PDF from: {pdf_link}")
                pdf_response = self._make_request(pdf_link, delay=1.5)  # Slightly longer delay for PDFs
                
                # Create PDF filename based on the same pattern as HTML
                pdf_filename = Path(file_path).stem + ".pdf"
                pdf_path = self.output_dir / doc_type / pdf_filename
                
                # Save PDF content
                with open(pdf_path, "wb") as f:
                    f.write(pdf_response.content)
                
                logger.info(f"PDF saved to: {pdf_path}")
                metadata["pdf_path"] = str(pdf_path)
                
                # Update metadata file with PDF path
                metadata_path = Path(file_path).with_suffix(".metadata.txt")
                with open(metadata_path, "a", encoding="utf-8") as f:
                    f.write(f"pdf_path: {pdf_path}\n")
                    
            except Exception as e:
                logger.error(f"Error downloading PDF: {e}")

        
        return file_path, metadata
    
    def scrape_circulars_by_year(self, year: int) -> List[Tuple[str, Dict]]:
        """Scrape circulars for a specific year.
        
        Args:
            year: Year to scrape circulars for
            
        Returns:
            List of (file_path, metadata) tuples
        """
        logger.info(f"Scraping circulars for year: {year}")
        
        # This is a placeholder implementation
        # In a real implementation, we would:
        # 1. Navigate to the circulars page
        # 2. Select the year from a dropdown or form
        # 3. Submit the form and get the results page
        # 4. Extract links to individual circulars
        # 5. Scrape each circular
        
        # For now, we'll return an empty list
        return []
    
    def scrape_notifications_by_year(self, year: int) -> List[Tuple[str, Dict]]:
        """Scrape notifications for a specific year.
        
        Args:
            year: Year to scrape notifications for
            
        Returns:
            List of (file_path, metadata) tuples
        """
        logger.info(f"Scraping notifications for year: {year}")
        
        # RBI Notifications page uses a form to filter by year/month
        # We need to simulate selecting the year and submitting
        
        # Initial request to get the form state (like __VIEWSTATE)
        try:
            response = self._make_request(self.NOTIFICATIONS_URL, delay=0.5)
            soup = BeautifulSoup(response.text, 'lxml')
            
            viewstate = soup.find('input', {'name': '__VIEWSTATE'})['value']
            eventvalidation = soup.find('input', {'name': '__EVENTVALIDATION'})['value']
            viewstategenerator = soup.find('input', {'name': '__VIEWSTATEGENERATOR'})['value']
            
        except Exception as e:
            logger.error(f"Failed to get initial form state for notifications: {e}")
            return []

        # Prepare form data for selecting the year
        form_data = {
            '__EVENTTARGET': '',
            '__EVENTARGUMENT': '',
            '__LASTFOCUS': '',
            '__VIEWSTATE': viewstate,
            '__VIEWSTATEGENERATOR': viewstategenerator,
            '__SCROLLPOSITIONX': '0',
            '__SCROLLPOSITIONY': '0',
            '__EVENTVALIDATION': eventvalidation,
            'drdnYear': str(year),  # Select the year
            'drdnMonth': '0',      # Select 'All' months
            'btnGo': 'Go'          # Simulate clicking the 'Go' button
        }
        
        scraped_docs = []
        
        try:
            # Post the form data to get results for the selected year
            logger.info(f"Fetching notification list for year {year}")
            # Add Referer header to mimic browser behavior
            post_headers = self.headers.copy()
            post_headers["Referer"] = self.NOTIFICATIONS_URL
            response = self.session.post(self.NOTIFICATIONS_URL, data=form_data, headers=post_headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Find the table containing the notification links
            # The exact selector might change if RBI updates their site structure
            # Inspecting the page (as of late 2024/early 2025) suggests 'table' within 'td.tabledata'
            table = soup.find('td', class_='tabledata').find('table')
            
            if not table:
                logger.warning(f"Could not find the notification table for year {year}")
                return []
                
            # Extract links from the table rows
            # Links are typically in the second column (index 1)
            links_found = 0
            for row in table.find_all('tr')[1:]: # Skip header row
                cols = row.find_all('td')
                if len(cols) > 1:
                    link_tag = cols[1].find('a', href=True)
                    if link_tag:
                        relative_url = link_tag['href']
                        # Construct absolute URL
                        if not relative_url.startswith('http'):
                           doc_url = f"{self.BASE_URL}/Scripts/{relative_url}"
                        else:
                           doc_url = relative_url # Should not happen based on observation, but handle just in case
                           
                        # Scrape the individual document
                        try:
                            file_path, metadata = self.scrape_document(doc_url, "notifications")
                            scraped_docs.append((file_path, metadata))
                            links_found += 1
                            # Add a small delay to avoid overwhelming the server
                            time.sleep(0.2) 
                        except Exception as e:
                           logger.error(f"Failed to scrape notification document {doc_url}: {e}")
            
            logger.info(f"Found and processed {links_found} notifications for year {year}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for year {year}: {e}")
        except Exception as e:
            logger.error(f"Error processing notifications for year {year}: {e}")

        return scraped_docs
    
    def scrape_all_documents(self, start_year: int = 2015, end_year: int = None) -> Dict[str, List[Tuple[str, Dict]]]:
        """Scrape all documents within a year range.
        
        Args:
            start_year: First year to scrape
            end_year: Last year to scrape (defaults to current year)
            
        Returns:
            Dictionary mapping document types to lists of (file_path, metadata) tuples
        """
        if end_year is None:
            end_year = datetime.now().year
        
        results = {
            "circulars": [],
            "notifications": []
        }
        
        logger.info(f"Scraping documents from {start_year} to {end_year}")
        
        for year in tqdm(range(start_year, end_year + 1), desc="Scraping years"):
            # Scrape circulars (Placeholder - needs implementation)
            logger.warning(f"Circular scraping for year {year} is not implemented yet.")
            # circulars = self.scrape_circulars_by_year(year) 
            # results["circulars"].extend(circulars)
            
            # Scrape notifications
            notifications = self.scrape_notifications_by_year(year)
            results["notifications"].extend(notifications)
            
            # Add a delay between years
            time.sleep(2) 
        
        logger.info(f"Completed scraping. Total notifications: {len(results['notifications'])}, Total circulars: {len(results['circulars'])}")
        return results


if __name__ == "__main__":
    # Example usage
    scraper = RBIScraper()
    
    # Scrape a single document (example URL)
    # In practice, you would get this URL from scraping the index pages
    example_url = "https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=12338"
    file_path, metadata = scraper.scrape_document(example_url, "notifications")
    
    print(f"Scraped document saved to: {file_path}")
    print(f"Metadata: {metadata}")