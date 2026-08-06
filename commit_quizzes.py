import os
import subprocess

for phase in ["21-java-android-foundations", "22-android-framework-system-basics"]:
    phase_dir = os.path.join("phases", phase)
    if not os.path.isdir(phase_dir):
        continue
        
    for lesson in sorted(os.listdir(phase_dir)):
        lesson_dir = os.path.join(phase_dir, lesson)
        if not os.path.isdir(lesson_dir):
            continue
            
        quiz_file = os.path.join(lesson_dir, "quiz.json")
        if os.path.exists(quiz_file):
            # check if there are changes
            status = subprocess.run(["git", "status", "--porcelain", quiz_file], capture_output=True, text=True).stdout
            if status.strip():
                subprocess.run(["git", "add", quiz_file])
                
                # Check if there are other things in the same lesson
                site_quiz_file = os.path.join("site", "phases", phase, lesson, "quiz.json")
                if os.path.exists(site_quiz_file):
                    subprocess.run(["git", "add", site_quiz_file])
                    
                commit_msg = f"feat(phase-{phase.split('-')[0]}/{lesson.split('-')[0]}): add extra quiz questions"
                subprocess.run(["git", "commit", "-m", commit_msg])

