import asyncio
import aiohttp
import numpy as np
import time

async def test_single_detection(session, url, flow_id):
    """Test single detection"""
    flow = np.random.randn(78).tolist()
    
    start = time.time()
    async with session.post(f"{url}/api/detect", json={"features": flow}) as resp:
        result = await resp.json()
        duration = (time.time() - start) * 1000
        
        print(f"Flow {flow_id}: {result.get('attack', 'None')} "
              f"(conf: {result.get('confidence', 0):.2f}) - {duration:.1f}ms")
        return result

async def test_batch_detection():
    """Test multiple concurrent detections"""
    url = "http://localhost:5000"
    
    async with aiohttp.ClientSession() as session:
        # Check health
        async with session.get(f"{url}/api/health") as resp:
            health = await resp.json()
            print(f"Health: {health}\n")
        
        # Learn a test attack first
        test_examples = [np.random.randn(78).tolist() for _ in range(3)]
        await session.post(f"{url}/api/learn", json={
            "attack_name": "TestAttack",
            "examples": test_examples
        })
        
        # Run concurrent detections
        tasks = [test_single_detection(session, url, i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        # Stats
        times = [r.get('processing_time_ms', 0) for r in results]
        print(f"\n📊 Batch Stats:")
        print(f"   Avg time: {sum(times)/len(times):.1f}ms")
        print(f"   Min time: {min(times):.1f}ms")
        print(f"   Max time: {max(times):.1f}ms")

async def main():
    print("🚀 Testing Async API\n")
    await test_batch_detection()

if __name__ == "__main__":
    asyncio.run(main())