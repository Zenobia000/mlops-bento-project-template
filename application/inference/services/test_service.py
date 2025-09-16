import requests
import json
import numpy as np
import time
from typing import Dict, List, Any

def test_numpy_endpoint():
    """Test NumPy endpoint"""
    print("🧪 Testing NumPy endpoint...")

    # Test data (Iris Setosa)
    test_data = np.array([[5.1, 3.5, 1.4, 0.2]])

    try:
        response = requests.post(
            "http://localhost:3000/classify",
            headers={"Content-Type": "application/json"},
            json=test_data.tolist()
        )

        if response.status_code == 200:
            result = response.json()
            print("✅ NumPy endpoint test successful")
            print(f"Prediction result: {json.dumps(result, indent=2, ensure_ascii=False)}")
            return result
        else:
            print(f"❌ NumPy endpoint test failed: {response.status_code}")
            print(response.text)
            return None

    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. Please ensure service is running.")
        print("Start with: poetry run bentoml serve iris_service:IrisClassifier --reload")
        return None
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return None

def test_json_endpoint():
    """Test JSON endpoint"""
    print("\n🧪 Testing JSON endpoint...")

    # Single prediction
    test_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    try:
        response = requests.post(
            "http://localhost:3000/classify_json",
            headers={"Content-Type": "application/json"},
            json=test_data
        )

        if response.status_code == 200:
            result = response.json()
            print("✅ JSON endpoint test successful")
            print(f"Prediction result: {json.dumps(result, indent=2, ensure_ascii=False)}")
            return result
        else:
            print(f"❌ JSON endpoint test failed: {response.status_code}")
            print(response.text)
            return None

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return None

def test_batch_endpoint():
    """Test batch prediction endpoint"""
    print("\n🧪 Testing batch prediction endpoint...")

    # Batch prediction
    test_data = {
        "instances": [
            {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},  # Setosa
            {"sepal_length": 6.2, "sepal_width": 2.9, "petal_length": 4.3, "petal_width": 1.3},  # Versicolor
            {"sepal_length": 7.3, "sepal_width": 2.9, "petal_length": 6.3, "petal_width": 1.8}   # Virginica
        ]
    }

    try:
        response = requests.post(
            "http://localhost:3000/classify_json",
            headers={"Content-Type": "application/json"},
            json=test_data
        )

        if response.status_code == 200:
            result = response.json()
            print("✅ Batch prediction test successful")
            print(f"Prediction result: {json.dumps(result, indent=2, ensure_ascii=False)}")
            return result
        else:
            print(f"❌ Batch prediction test failed: {response.status_code}")
            print(response.text)
            return None

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return None

def test_health_check():
    """Test health check endpoint"""
    print("\n🧪 Testing health check endpoint...")

    try:
        response = requests.get("http://localhost:3000/health_check")

        if response.status_code == 200:
            result = response.json()
            print("✅ Health check test successful")
            print(f"Service status: {json.dumps(result, indent=2, ensure_ascii=False)}")
            return result
        else:
            print(f"❌ Health check test failed: {response.status_code}")
            print(response.text)
            return None

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return None

def test_metrics_endpoint():
    """Test metrics summary endpoint"""
    print("\n🧪 Testing metrics summary endpoint...")

    try:
        response = requests.get("http://localhost:3000/get_metrics_summary?hours=1")

        if response.status_code == 200:
            result = response.json()
            print("✅ Metrics endpoint test successful")
            print(f"Metrics summary: {json.dumps(result, indent=2, ensure_ascii=False)}")
            return result
        else:
            print(f"❌ Metrics endpoint test failed: {response.status_code}")
            print(response.text)
            return None

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return None

def test_feedback_endpoint():
    """Test feedback endpoint"""
    print("\n🧪 Testing feedback endpoint...")

    # Sample feedback data
    feedback_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
        "predicted_class": "setosa",
        "actual_class": "setosa",
        "confidence": 0.95
    }

    try:
        response = requests.post(
            "http://localhost:3000/feedback",
            headers={"Content-Type": "application/json"},
            json=feedback_data
        )

        if response.status_code == 200:
            result = response.json()
            print("✅ Feedback endpoint test successful")
            print(f"Feedback result: {json.dumps(result, indent=2, ensure_ascii=False)}")
            return result
        else:
            print(f"❌ Feedback endpoint test failed: {response.status_code}")
            print(response.text)
            return None

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return None

def run_performance_test(num_requests: int = 100):
    """Run basic performance test"""
    print(f"\n⚡ Running performance test ({num_requests} requests)...")

    test_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response_times = []
    successful_requests = 0
    failed_requests = 0

    start_time = time.time()

    for i in range(num_requests):
        try:
            request_start = time.time()
            response = requests.post(
                "http://localhost:3000/classify_json",
                headers={"Content-Type": "application/json"},
                json=test_data,
                timeout=5
            )
            request_end = time.time()

            if response.status_code == 200:
                successful_requests += 1
                response_times.append((request_end - request_start) * 1000)  # Convert to ms
            else:
                failed_requests += 1

        except Exception as e:
            failed_requests += 1

        # Progress indicator
        if (i + 1) % 20 == 0:
            print(f"  Completed {i + 1}/{num_requests} requests...")

    total_time = time.time() - start_time

    # Calculate statistics
    if response_times:
        avg_response_time = np.mean(response_times)
        min_response_time = np.min(response_times)
        max_response_time = np.max(response_times)
        p95_response_time = np.percentile(response_times, 95)
        requests_per_second = successful_requests / total_time

        print("📊 Performance Test Results:")
        print(f"  Total requests: {num_requests}")
        print(f"  Successful: {successful_requests}")
        print(f"  Failed: {failed_requests}")
        print(f"  Success rate: {(successful_requests / num_requests) * 100:.1f}%")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Requests per second: {requests_per_second:.2f}")
        print(f"  Average response time: {avg_response_time:.2f} ms")
        print(f"  Min response time: {min_response_time:.2f} ms")
        print(f"  Max response time: {max_response_time:.2f} ms")
        print(f"  95th percentile: {p95_response_time:.2f} ms")

        return {
            "total_requests": num_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate": (successful_requests / num_requests) * 100,
            "total_time_seconds": total_time,
            "requests_per_second": requests_per_second,
            "avg_response_time_ms": avg_response_time,
            "min_response_time_ms": min_response_time,
            "max_response_time_ms": max_response_time,
            "p95_response_time_ms": p95_response_time
        }
    else:
        print("❌ No successful requests for performance analysis")
        return None

def validate_prediction_format(prediction_result: Dict[str, Any]) -> bool:
    """Validate prediction result format"""
    if not prediction_result:
        return False

    required_keys = ["predictions"]
    if not all(key in prediction_result for key in required_keys):
        print(f"❌ Missing required keys: {required_keys}")
        return False

    predictions = prediction_result["predictions"]
    if not isinstance(predictions, list) or len(predictions) == 0:
        print("❌ Predictions should be a non-empty list")
        return False

    # Check first prediction format
    first_pred = predictions[0]
    required_pred_keys = ["prediction", "prediction_id", "confidence", "probabilities"]
    if not all(key in first_pred for key in required_pred_keys):
        print(f"❌ Prediction missing required keys: {required_pred_keys}")
        return False

    # Validate prediction classes
    valid_classes = ["setosa", "versicolor", "virginica"]
    if first_pred["prediction"] not in valid_classes:
        print(f"❌ Invalid prediction class: {first_pred['prediction']}")
        return False

    # Validate confidence is between 0 and 1
    if not (0 <= first_pred["confidence"] <= 1):
        print(f"❌ Invalid confidence score: {first_pred['confidence']}")
        return False

    print("✅ Prediction format validation passed")
    return True

def main():
    """Run all tests"""
    print("🚀 Starting Iris Classification Service Tests...")
    print("Please ensure service is running: poetry run bentoml serve iris_service:IrisClassifier --reload")
    print("-" * 80)

    # Wait a moment for user to see the message
    time.sleep(2)

    test_results = {}

    # Basic functionality tests
    print("📋 Running Basic Functionality Tests...\n")

    health_result = test_health_check()
    test_results['health_check'] = health_result

    numpy_result = test_numpy_endpoint()
    test_results['numpy_endpoint'] = numpy_result
    if numpy_result:
        validate_prediction_format(numpy_result)

    json_result = test_json_endpoint()
    test_results['json_endpoint'] = json_result
    if json_result:
        validate_prediction_format(json_result)

    batch_result = test_batch_endpoint()
    test_results['batch_endpoint'] = batch_result
    if batch_result:
        validate_prediction_format(batch_result)

    metrics_result = test_metrics_endpoint()
    test_results['metrics_endpoint'] = metrics_result

    feedback_result = test_feedback_endpoint()
    test_results['feedback_endpoint'] = feedback_result

    # Performance test
    print("\n" + "="*80)
    print("📈 Running Performance Tests...\n")
    perf_result = run_performance_test(50)  # Smaller number for quicker testing
    test_results['performance_test'] = perf_result

    # Summary
    print("\n" + "="*80)
    print("📋 Test Summary:")
    successful_tests = sum(1 for result in test_results.values() if result is not None)
    total_tests = len(test_results)
    print(f"  Successful tests: {successful_tests}/{total_tests}")

    if successful_tests == total_tests:
        print("🎉 All tests passed! Service is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the logs above.")

    return test_results

if __name__ == "__main__":
    results = main()