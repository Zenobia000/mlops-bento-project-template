#!/usr/bin/env python3
"""
負載測試和性能測試腳本
用於測試 BentoML 服務的性能和穩定性
"""

import asyncio
import aiohttp
import time
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import statistics
from datetime import datetime
import argparse

class LoadTester:
    """負載測試器"""

    def __init__(self, base_url="http://localhost:3000"):
        self.base_url = base_url
        self.results = {
            "start_time": None,
            "end_time": None,
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "response_times": [],
            "errors": []
        }

    def generate_test_data(self):
        """生成測試資料"""
        # 生成隨機的 Iris 特徵資料
        return {
            "sepal_length": np.random.uniform(4.0, 8.0),
            "sepal_width": np.random.uniform(2.0, 5.0),
            "petal_length": np.random.uniform(1.0, 7.0),
            "petal_width": np.random.uniform(0.1, 3.0)
        }

    async def make_single_request(self, session, endpoint="/classify_json"):
        """發送單個請求"""
        test_data = self.generate_test_data()

        start_time = time.time()
        try:
            async with session.post(
                f"{self.base_url}{endpoint}",
                json=test_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                end_time = time.time()
                response_time = (end_time - start_time) * 1000  # 轉換為毫秒

                if response.status == 200:
                    await response.json()  # 讀取響應
                    self.results["successful_requests"] += 1
                    self.results["response_times"].append(response_time)
                    return True, response_time
                else:
                    error_msg = f"HTTP {response.status}"
                    self.results["errors"].append(error_msg)
                    self.results["failed_requests"] += 1
                    return False, response_time

        except Exception as e:
            end_time = time.time()
            response_time = (end_time - start_time) * 1000
            error_msg = f"Exception: {str(e)}"
            self.results["errors"].append(error_msg)
            self.results["failed_requests"] += 1
            return False, response_time

    async def run_concurrent_test(self, concurrent_users=10, requests_per_user=10):
        """運行並發測試"""
        print(f"🚀 開始並發負載測試:")
        print(f"   並發用戶: {concurrent_users}")
        print(f"   每用戶請求數: {requests_per_user}")
        print(f"   總請求數: {concurrent_users * requests_per_user}")
        print("-" * 60)

        self.results["start_time"] = datetime.now()
        self.results["total_requests"] = concurrent_users * requests_per_user

        connector = aiohttp.TCPConnector(limit=100, limit_per_host=50)
        timeout = aiohttp.ClientTimeout(total=30)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # 創建所有請求的任務
            tasks = []
            for user in range(concurrent_users):
                for request in range(requests_per_user):
                    task = self.make_single_request(session)
                    tasks.append(task)

            # 執行所有請求
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()

            self.results["end_time"] = datetime.now()
            total_time = end_time - start_time

            print(f"✅ 測試完成!")
            print(f"   總耗時: {total_time:.2f} 秒")
            print(f"   成功請求: {self.results['successful_requests']}")
            print(f"   失敗請求: {self.results['failed_requests']}")

            return self.analyze_results(total_time)

    def run_sequential_test(self, num_requests=100):
        """運行順序測試"""
        print(f"🚀 開始順序負載測試:")
        print(f"   請求數量: {num_requests}")
        print("-" * 60)

        self.results["start_time"] = datetime.now()
        self.results["total_requests"] = num_requests

        import requests

        start_time = time.time()

        for i in range(num_requests):
            test_data = self.generate_test_data()

            request_start = time.time()
            try:
                response = requests.post(
                    f"{self.base_url}/classify_json",
                    json=test_data,
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
                request_end = time.time()
                response_time = (request_end - request_start) * 1000

                if response.status_code == 200:
                    self.results["successful_requests"] += 1
                    self.results["response_times"].append(response_time)
                else:
                    self.results["failed_requests"] += 1
                    self.results["errors"].append(f"HTTP {response.status_code}")

            except Exception as e:
                request_end = time.time()
                self.results["failed_requests"] += 1
                self.results["errors"].append(str(e))

            # 進度顯示
            if (i + 1) % 10 == 0:
                print(f"   進度: {i + 1}/{num_requests}")

        end_time = time.time()
        total_time = end_time - start_time

        self.results["end_time"] = datetime.now()

        print(f"✅ 測試完成!")
        print(f"   總耗時: {total_time:.2f} 秒")
        print(f"   成功請求: {self.results['successful_requests']}")
        print(f"   失敗請求: {self.results['failed_requests']}")

        return self.analyze_results(total_time)

    def analyze_results(self, total_time):
        """分析測試結果"""
        if not self.results["response_times"]:
            print("❌ 沒有成功的請求來分析")
            return None

        response_times = self.results["response_times"]

        analysis = {
            "performance_metrics": {
                "total_time_seconds": total_time,
                "total_requests": self.results["total_requests"],
                "successful_requests": self.results["successful_requests"],
                "failed_requests": self.results["failed_requests"],
                "success_rate_percent": (self.results["successful_requests"] / self.results["total_requests"]) * 100,
                "requests_per_second": self.results["total_requests"] / total_time,
                "successful_requests_per_second": self.results["successful_requests"] / total_time
            },
            "response_time_metrics": {
                "min_ms": min(response_times),
                "max_ms": max(response_times),
                "mean_ms": statistics.mean(response_times),
                "median_ms": statistics.median(response_times),
                "p95_ms": np.percentile(response_times, 95),
                "p99_ms": np.percentile(response_times, 99),
                "std_dev_ms": statistics.stdev(response_times) if len(response_times) > 1 else 0
            },
            "error_analysis": {
                "total_errors": len(self.results["errors"]),
                "unique_errors": len(set(self.results["errors"])),
                "error_types": {}
            }
        }

        # 分析錯誤類型
        for error in self.results["errors"]:
            if error in analysis["error_analysis"]["error_types"]:
                analysis["error_analysis"]["error_types"][error] += 1
            else:
                analysis["error_analysis"]["error_types"][error] = 1

        # 打印分析結果
        print("\n📊 性能分析:")
        print("-" * 60)
        perf = analysis["performance_metrics"]
        print(f"總請求數: {perf['total_requests']}")
        print(f"成功率: {perf['success_rate_percent']:.1f}%")
        print(f"每秒請求數 (總): {perf['requests_per_second']:.1f}")
        print(f"每秒請求數 (成功): {perf['successful_requests_per_second']:.1f}")

        print("\n⏱️ 響應時間分析:")
        print("-" * 60)
        resp = analysis["response_time_metrics"]
        print(f"最小響應時間: {resp['min_ms']:.1f} ms")
        print(f"最大響應時間: {resp['max_ms']:.1f} ms")
        print(f"平均響應時間: {resp['mean_ms']:.1f} ms")
        print(f"中位數響應時間: {resp['median_ms']:.1f} ms")
        print(f"95th 百分位: {resp['p95_ms']:.1f} ms")
        print(f"99th 百分位: {resp['p99_ms']:.1f} ms")

        if analysis["error_analysis"]["total_errors"] > 0:
            print("\n❌ 錯誤分析:")
            print("-" * 60)
            for error_type, count in analysis["error_analysis"]["error_types"].items():
                print(f"{error_type}: {count} 次")

        return analysis

    def save_results(self, analysis, filename=None):
        """儲存測試結果"""
        if filename is None:
            filename = f"load_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        results_data = {
            "test_configuration": {
                "base_url": self.base_url,
                "start_time": self.results["start_time"].isoformat(),
                "end_time": self.results["end_time"].isoformat()
            },
            "raw_results": self.results,
            "analysis": analysis
        }

        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)

        print(f"\n💾 測試結果已儲存到: {filename}")
        return filename

def check_service_health(base_url):
    """檢查服務健康狀態"""
    import requests

    try:
        response = requests.get(f"{base_url}/health_check", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ 服務健康檢查通過")
            print(f"   狀態: {health_data.get('status', 'unknown')}")
            print(f"   模型: {health_data.get('model_name', 'unknown')}")
            return True
        else:
            print(f"❌ 服務健康檢查失敗: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 無法連接到服務: {str(e)}")
        return False

async def main():
    parser = argparse.ArgumentParser(description="BentoML 服務負載測試")
    parser.add_argument("--url", type=str, default="http://localhost:3000",
                       help="服務基礎 URL")
    parser.add_argument("--concurrent", type=int, default=10,
                       help="並發用戶數")
    parser.add_argument("--requests", type=int, default=10,
                       help="每用戶請求數")
    parser.add_argument("--sequential", action="store_true",
                       help="運行順序測試而不是並發測試")
    parser.add_argument("--output", type=str,
                       help="結果輸出文件名")

    args = parser.parse_args()

    print("🧪 BentoML 服務負載測試")
    print("=" * 60)

    # 檢查服務健康狀態
    if not check_service_health(args.url):
        print("\n請確保 BentoML 服務正在運行:")
        print("poetry run bentoml serve application/inference/services/iris_service:IrisClassifier --reload")
        return

    # 創建負載測試器
    tester = LoadTester(base_url=args.url)

    # 運行測試
    if args.sequential:
        total_requests = args.concurrent * args.requests
        analysis = tester.run_sequential_test(num_requests=total_requests)
    else:
        analysis = await tester.run_concurrent_test(
            concurrent_users=args.concurrent,
            requests_per_user=args.requests
        )

    # 保存結果
    if analysis:
        tester.save_results(analysis, args.output)

    # 性能建議
    if analysis:
        perf = analysis["performance_metrics"]
        resp = analysis["response_time_metrics"]

        print("\n💡 性能建議:")
        print("-" * 60)

        if perf["success_rate_percent"] < 95:
            print("⚠️ 成功率偏低，檢查錯誤日誌和服務配置")

        if resp["mean_ms"] > 100:
            print("⚠️ 平均響應時間較長，考慮優化模型或增加資源")

        if resp["p95_ms"] > 200:
            print("⚠️ 95th 百分位響應時間較長，檢查服務穩定性")

        if perf["successful_requests_per_second"] < 10:
            print("⚠️ 吞吐量較低，考慮橫向擴展")

if __name__ == "__main__":
    asyncio.run(main())