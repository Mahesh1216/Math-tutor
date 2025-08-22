#!/usr/bin/env python3
"""
Standalone MCP Server for Math Agent Testing
This script can be run independently to test MCP functionality
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add the current directory to Python path to import mcp_client
sys.path.append(str(Path(__file__).parent))

from mcp_client import get_mcp_client, search_with_mcp, test_mcp_functionality
from dotenv import load_dotenv

async def interactive_test():
    """Interactive test of MCP functionality"""
    print("🧮 Math Agent MCP Server Interactive Test")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Check if required API keys are available
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        print("❌ TAVILY_API_KEY not found in environment variables")
        print("Please add your Tavily API key to the .env file")
        return
    
    print(f"✅ Found Tavily API Key: {tavily_key[:10]}...")
    
    # Initialize MCP client
    try:
        client = await get_mcp_client()
        print("✅ MCP Client initialized successfully")
        
        # List available tools
        tools = await client.list_tools()
        print(f"\n🔧 Available Tools:")
        for server_name, tool_list in tools.items():
            print(f"  Server: {server_name}")
            for tool in tool_list:
                print(f"    - {tool.name}: {tool.description}")
        
        # List available resources
        resources = await client.list_resources()
        print(f"\n📚 Available Resources:")
        for server_name, resource_list in resources.items():
            print(f"  Server: {server_name}")
            for resource in resource_list:
                print(f"    - {resource.name}: {resource.description}")
        
    except Exception as e:
        print(f"❌ Failed to initialize MCP client: {e}")
        return
    
    # Interactive search loop
    print("\n" + "="*50)
    print("🔍 Interactive Search Mode")
    print("Enter math questions to search for (or 'quit' to exit)")
    print("="*50)
    
    while True:
        try:
            question = input("\n📝 Enter your math question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question:
                continue
            
            print(f"\n🔍 Searching for: {question}")
            print("-" * 30)
            
            # Perform search using MCP
            results = await search_with_mcp(question, max_results=2)
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"\nResult {i}:")
                    print(f"Source: {result.get('source', 'Unknown')}")
                    content = result.get('content', 'No content')
                    # Truncate long content for better readability
                    if len(content) > 500:
                        content = content[:500] + "..."
                    print(f"Content: {content}")
                    print("-" * 30)
            else:
                print("❌ No results found")
                
        except KeyboardInterrupt:
            print("\n\n👋 Exiting...")
            break
        except Exception as e:
            print(f"❌ Error during search: {e}")

async def batch_test():
    """Test with a batch of predefined math questions"""
    print("🧪 Running Batch Test with Math Questions")
    print("=" * 50)
    
    # Load environment
    load_dotenv()
    
    # Test questions
    test_questions = [
        "What is the derivative of x^2?",
        "How to solve quadratic equations?",
        "What is the Pythagorean theorem?",
        "Explain integration by parts",
        "What is a complex number?"
    ]
    
    try:
        # Run basic functionality test first
        success = await test_mcp_functionality()
        if not success:
            print("❌ Basic MCP test failed")
            return
        
        print(f"\n🔍 Testing {len(test_questions)} math questions:")
        print("-" * 50)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{i}. Testing: {question}")
            
            try:
                results = await search_with_mcp(question, max_results=1)
                
                if results and results[0].get('content'):
                    content_preview = results[0]['content'][:200] + "..."
                    print(f"   ✅ Success: {content_preview}")
                else:
                    print(f"   ❌ No results found")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
            
            # Small delay between requests
            await asyncio.sleep(1)
        
        print(f"\n✅ Batch test completed!")
        
    except Exception as e:
        print(f"❌ Batch test failed: {e}")

def print_usage():
    """Print usage information"""
    print("📖 MCP Server Test Usage:")
    print("  python mcp_server.py interactive  # Interactive testing mode")
    print("  python mcp_server.py batch       # Batch testing mode")
    print("  python mcp_server.py test        # Basic functionality test")
    print("  python mcp_server.py             # Show this help")

async def main():
    """Main function"""
    if len(sys.argv) < 2:
        print_usage()
        return
    
    mode = sys.argv[1].lower()
    
    if mode == "interactive":
        await interactive_test()
    elif mode == "batch":
        await batch_test()
    elif mode == "test":
        success = await test_mcp_functionality()
        if success:
            print("✅ MCP functionality test passed!")
        else:
            print("❌ MCP functionality test failed!")
    else:
        print(f"❌ Unknown mode: {mode}")
        print_usage()

if __name__ == "__main__":
    asyncio.run(main())