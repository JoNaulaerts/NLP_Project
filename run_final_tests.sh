#!/bin/bash
# Run FINAL COMPREHENSIVE test suite

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║        RUNNING FINAL COMPREHENSIVE TEST SUITE                      ║"
echo "║             This tests EVERYTHING (no skips)                       ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "⏱️  This will take 2-5 minutes (includes full RAG & Quiz tests)"
echo ""

# Copy test file into container
docker cp test_final_complete.py ml_learning_assistant_app:/app/test_final_complete.py

# Run the tests
docker exec -it ml_learning_assistant_app python3 /app/test_final_complete.py

echo ""
echo "✅ Test execution complete!"
echo ""
echo "📄 View detailed results:"
echo "   docker exec ml_learning_assistant_app cat /app/data/test_results/final_test_results_*.json"
