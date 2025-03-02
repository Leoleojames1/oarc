"""Enhanced DuckDuckGo Search API with storage capabilities.
Handles text, image, and news searches with results persistence.
"""
from duckduckgo_search import DDGS
import json
import aiohttp
import asyncio
from datetime import datetime
from pathlib import Path
import pymongo
import logging
from typing import List, Dict, Optional, Union

class DuckDuckGoSearchAPI:
    def __init__(self, storage_type: str = "json", database_path: str = "searchResults.db"):
        """
        Initialize the DuckDuckGo Search API wrapper.
        
        Args:
            storage_type: Type of storage ("json" or "sqlite")
            database_path: Path to the database file
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

    async def text_search(self, search_query: str, max_results: int = 10) -> List[Dict]:
        """
        Perform an async text search using DuckDuckGo.
        
        Args:
            search_query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of formatted search results
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"https://duckduckgo.com/?q={search_query}&format=json&pretty=1") as response:
                    results = await response.json()
                    formatted_results = self.format_results(results)
                    await self.store_results(search_query, "text", formatted_results)
                    return formatted_results
        except Exception as e:
            self.logger.error(f"Text search error: {e}")
            raise

    async def image_search(self, search_query: str, max_results: int = 10) -> List[Dict]:
        """Perform an async image search using DuckDuckGo."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"https://duckduckgo.com/?q={search_query}&format=json&pretty=1&iax=images&ia=images") as response:
                    results = await response.json()
                    formatted_results = self.format_image_results(results)
                    await self.store_results(search_query, "image", formatted_results)
                    return formatted_results
        except Exception as e:
            self.logger.error(f"Image search error: {e}")
            raise

    async def news_search(self, search_query: str, max_results: int = 20) -> List[Dict]:
        """Perform an async news search using DuckDuckGo."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"https://duckduckgo.com/?q={search_query}&format=json&pretty=1&ia=news") as response:
                    results = await response.json()
                    formatted_results = self.format_news_results(results)
                    await self.store_results(search_query, "news", formatted_results)
                    return formatted_results
        except Exception as e:
            self.logger.error(f"News search error: {e}")
            raise

    def format_results(self, results: List[Dict]) -> List[Dict]:
        """Format text search results."""
        return [{
            "title": result.get("title"),
            "link": result.get("link"),
            "snippet": result.get("body"),
            "source": result.get("source"),
            "timestamp": datetime.now().isoformat()
        } for result in results]

    def format_image_results(self, results: List[Dict]) -> List[Dict]:
        """Format image search results."""
        return [{
            "title": result.get("title"),
            "image_url": result.get("image"),
            "thumbnail": result.get("thumbnail"),
            "source_url": result.get("url"),
            "timestamp": datetime.now().isoformat()
        } for result in results]

    def format_news_results(self, results: List[Dict]) -> List[Dict]:
        """Format news search results."""
        return [{
            "title": result.get("title"),
            "link": result.get("link"),
            "snippet": result.get("excerpt"),
            "published": result.get("date"),
            "source": result.get("source"),
            "timestamp": datetime.now().isoformat()
        } for result in results]

    async def store_results(self, search_query: str, search_type: str, results: List[Dict]) -> None:
        """Store search results in the database."""
        client = pymongo.MongoClient(self.database_path)
        db = client.searchResults
        collection = db[search_type]
        for result in results:
            result["search_query"] = search_query
        collection.insert_many(results)
        client.close()