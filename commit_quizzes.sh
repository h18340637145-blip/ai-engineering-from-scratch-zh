#!/bin/bash
# First commit the AGENTS.md update
if git status --porcelain AGENTS.md | grep -q "M"; then
  git add AGENTS.md
  git commit -m "chore: update quiz.json schema rule to allow 6-9 questions"
fi

for p in phases/21-java-android-foundations/* phases/22-android-framework-system-basics/*; do
  if [ -d "$p" ]; then
    if git status --porcelain "$p/quiz.json" | grep -q "M"; then
      phase=$(echo "$p" | cut -d'/' -f2 | cut -d'-' -f1)
      lesson=$(echo "$p" | cut -d'/' -f3 | cut -d'-' -f1)
      
      git add "$p/quiz.json"
      
      site_quiz="site/$p/quiz.json"
      if [ -f "$site_quiz" ]; then
        git add "$site_quiz"
      fi
      
      git commit -m "feat(phase-$phase/$lesson): add extra quiz questions"
    fi
  fi
done

# Finally, commit the site data
if git status --porcelain site/data.js | grep -q "M"; then
  git add site/data.js
  git commit -m "chore(site): rebuild data.js"
fi
