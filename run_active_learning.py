from active_learning.loop import ActiveLearningLoop

if __name__ == "__main__":
    al = ActiveLearningLoop()
    al.run(iterations=5)