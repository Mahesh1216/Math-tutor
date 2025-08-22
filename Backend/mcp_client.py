"""
Custom MCP (Model Context Protocol) Implementation for Math Agent
This implements MCP concepts without requiring the official SDK
"""

import asyncio
import json
import logging
import os
import httpx
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MCPTool:
    """Represents an MCP tool following the official specification"""
    name: str
    description: str
    inputSchema: Dict[str, Any]

@dataclass
class MCPResource:
    """Represents an MCP resource following the official specification"""
    uri: str
    name: str
    description: str
    mimeType: str

@dataclass
class MCPTextContent:
    """Represents MCP text content"""
    type: str = "text"
    text: str = ""

class MCPServer(ABC):
    """Abstract base class for MCP servers"""
    
    def __init__(self, name: str):
        self.name = name
        self.tools: List[MCPTool] = []
        self.resources: List[MCPResource] = []
        self._session: Optional[httpx.AsyncClient] = None
    
    @abstractmethod
    async def list_tools(self) -> List[MCPTool]:
        """List available tools"""
        pass
    
    @abstractmethod
    async def list_resources(self) -> List[MCPResource]:
        """List available resources"""
        pass
    
    @abstractmethod
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> List[MCPTextContent]:
        """Call a specific tool"""
        pass
    
    async def cleanup(self):
        """Cleanup resources"""
        if self._session:
            await self._session.aclose()
            self._session = None

class WebSearchMCPServer(MCPServer):
    """MCP Server for web search using Tavily API"""
    
    def __init__(self, tavily_api_key: str):
        super().__init__("web-search-mcp-server")
        self.tavily_api_key = tavily_api_key
        
        # Define available tools
        self.tools = [
            MCPTool(
                name="web_search",
                description="Search the web for mathematical information using Tavily API",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query for mathematical information"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of search results to return",
                            "default": 3,
                            "minimum": 1,
                            "maximum": 10
                        }
                    },
                    "required": ["query"]
                }
            )
        ]
        
        # Define available resources
        self.resources = [
            MCPResource(
                uri="search://web/math",
                name="Mathematical Web Search",
                description="Web search capabilities specialized for mathematical content",
                mimeType="application/json"
            )
        ]
    
    async def list_tools(self) -> List[MCPTool]:
        """List available tools"""
        return self.tools
    
    async def list_resources(self) -> List[MCPResource]:
        """List available resources"""
        return self.resources
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> List[MCPTextContent]:
        """Call a specific tool"""
        if name == "web_search":
            return await self._perform_web_search(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")
    
    async def _get_session(self) -> httpx.AsyncClient:
        """Get or create HTTP session"""
        if not self._session:
            self._session = httpx.AsyncClient(timeout=30.0)
        return self._session
    
    async def _perform_web_search(self, arguments: Dict[str, Any]) -> List[MCPTextContent]:
        """Perform web search using Tavily API"""
        query = arguments.get("query")
        max_results = arguments.get("max_results", 3)
        
        if not query:
            return [MCPTextContent(
                type="text",
                text="Error: Query parameter is required"
            )]
        
        headers = {"Content-Type": "application/json"}
        payload = {
            "api_key": self.tavily_api_key,
            "query": f"mathematics {query}",  # Add math context
            "search_depth": "basic",
            "include_answer": True,
            "include_images": False,
            "include_raw_content": False,
            "max_results": max_results,
        }
        
        try:
            session = await self._get_session()
            response = await session.post(
                "https://api.tavily.com/search",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            # Format response according to MCP standards
            content_parts = []
            
            # Add direct answer if available
            if data.get("answer"):
                content_parts.append(f"🎯 Direct Answer:\n{data['answer']}\n")
            
            # Add search results
            results = data.get("results", [])
            if results:
                content_parts.append("🔍 Search Results:\n")
                for i, result in enumerate(results[:max_results], 1):
                    title = result.get("title", "No title")
                    content = result.get("content", "No content available")
                    url = result.get("url", "No URL")
                    score = result.get("score", 0)
                    
                    content_parts.append(f"\n{i}. **{title}**")
                    content_parts.append(f"   📊 Relevance Score: {score:.2f}")
                    content_parts.append(f"   📄 Content: {content[:400]}...")
                    content_parts.append(f"   🔗 URL: {url}\n")
            
            final_content = "\n".join(content_parts) if content_parts else "No search results found."
            
            return [MCPTextContent(
                type="text",
                text=final_content
            )]
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error during web search: {e}")
            return [MCPTextContent(
                type="text",
                text=f"HTTP Error: {str(e)}"
            )]
        except Exception as e:
            logger.error(f"Unexpected error during web search: {e}")
            return [MCPTextContent(
                type="text",
                text=f"Search Error: {str(e)}"
            )]

class MCPClient:
    """MCP Client that communicates with MCP servers"""
    
    def __init__(self):
        self.servers: Dict[str, MCPServer] = {}
    
    def add_server(self, server_name: str, server: MCPServer):
        """Add an MCP server to the client"""
        self.servers[server_name] = server
        logger.info(f"Added MCP server: {server_name}")
    
    async def list_tools(self, server_name: Optional[str] = None) -> Dict[str, List[MCPTool]]:
        """List tools from all servers or a specific server"""
        if server_name:
            if server_name not in self.servers:
                raise ValueError(f"Server '{server_name}' not found")
            return {server_name: await self.servers[server_name].list_tools()}
        
        tools_by_server = {}
        for name, server in self.servers.items():
            tools_by_server[name] = await server.list_tools()
        return tools_by_server
    
    async def list_resources(self, server_name: Optional[str] = None) -> Dict[str, List[MCPResource]]:
        """List resources from all servers or a specific server"""
        if server_name:
            if server_name not in self.servers:
                raise ValueError(f"Server '{server_name}' not found")
            return {server_name: await self.servers[server_name].list_resources()}
        
        resources_by_server = {}
        for name, server in self.servers.items():
            resources_by_server[name] = await server.list_resources()
        return resources_by_server
    
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> List[MCPTextContent]:
        """Call a tool on a specific server"""
        if server_name not in self.servers:
            raise ValueError(f"Server '{server_name}' not found")
        
        return await self.servers[server_name].call_tool(tool_name, arguments)
    
    async def cleanup(self):
        """Cleanup all servers"""
        for server in self.servers.values():
            await server.cleanup()

# Global MCP client instance
_mcp_client: Optional[MCPClient] = None

async def get_mcp_client() -> MCPClient:
    """Get or create the global MCP client"""
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MCPClient()
        
        # Initialize web search server
        tavily_key = os.getenv("TAVILY_API_KEY")
        if tavily_key:
            web_search_server = WebSearchMCPServer(tavily_key)
            _mcp_client.add_server("web_search", web_search_server)
            logger.info("MCP client initialized with web search server")
        else:
            logger.error("TAVILY_API_KEY not found - MCP web search will not be available")
    
    return _mcp_client

async def search_with_mcp(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """
    Convenience function to perform web search using MCP
    Returns results in format compatible with your existing code
    """
    try:
        client = await get_mcp_client()
        
        # Call the web search tool
        results = await client.call_tool(
            "web_search", 
            "web_search", 
            {"query": query, "max_results": max_results}
        )
        
        # Convert MCP results to your expected format
        formatted_results = []
        for result in results:
            formatted_results.append({
                "source": "MCP Web Search",
                "content": result.text
            })
        
        return formatted_results if formatted_results else [
            {"source": "MCP Web Search", "content": "No results found"}
        ]
        
    except Exception as e:
        logger.error(f"MCP search error: {e}")
        return [{"source": "MCP Error", "content": f"Search failed: {str(e)}"}]

async def cleanup_mcp():
    """Cleanup MCP resources"""
    global _mcp_client
    if _mcp_client:
        await _mcp_client.cleanup()
        _mcp_client = None
        logger.info("MCP client cleaned up")

async def test_mcp_functionality():
    """Test function to verify MCP is working"""
    print("🧪 Testing Custom MCP Implementation...")
    
    try:
        client = await get_mcp_client()
        
        # Test listing tools
        tools = await client.list_tools()
        print(f"✅ Available tools: {list(tools.keys())}")
        
        # Test web search
        results = await search_with_mcp("What is calculus?", max_results=2)
        print(f"✅ Search test successful - got {len(results)} results")
        
        if results and results[0].get("content"):
            preview = results[0]["content"][:200] + "..."
            print(f"📄 Sample result: {preview}")
        
        return True
        
    except Exception as e:
        print(f"❌ MCP test failed: {e}")
        return False

if __name__ == "__main__":
    # Test the MCP implementation
    import asyncio
    from dotenv import load_dotenv
    
    load_dotenv()
    asyncio.run(test_mcp_functionality())