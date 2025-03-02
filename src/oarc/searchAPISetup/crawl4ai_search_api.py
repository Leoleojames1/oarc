import asyncio
from crawl4ai import AsyncWebCrawler
import pymongo
import logging
from datetime import datetime
from typing import List, Dict

class Crawl4AISearchAPI:
    def __init__(self, storage_type: str = "mongodb", database_path: str = "mongodb://localhost:27017/"):
        """
        Initialize the Crawl4AI Search API wrapper.
        
        Args:
            storage_type: Type of storage ("mongodb")
            database_path: Path to the database file or MongoDB URI
        """
        self.storage_type = storage_type
        self.database_path = database_path
        self.setup_logging()
    
    def setup_logging(self):
        """Configure logging for the API."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    async def scrape_url(self, url: str) -> Dict:
        """
        Perform an async web scrape using Crawl4AI.
        
        Args:
            url: URL to scrape
            
        Returns:
            Formatted scrape result
        """
        try:
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=url)
                formatted_result = self.format_result(result)
                await self.store_result(url, formatted_result)
                return formatted_result
        except Exception as e:
            self.logger.error(f"Scrape error: {e}")
            raise

    def format_result(self, result: Dict) -> Dict:
        """Format scrape result."""
        return {
            "title": result.get("title"),
            "content": result.get("markdown"),
            "url": result.get("url"),
            "timestamp": datetime.now().isoformat()
        }

    async def store_result(self, url: str, result: Dict) -> None:
        """Store scrape result in the database."""
        if self.storage_type == "mongodb":
            client = pymongo.MongoClient(self.database_path)
            db = client.scrapeResults
            collection = db["web_scrape"]
            result["search_query"] = url
            collection.insert_one(result)
            client.close()
        else:
            # Handle other storage types if needed
            pass