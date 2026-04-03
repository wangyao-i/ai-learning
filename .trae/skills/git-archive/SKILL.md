---
name: "git-archive"
description: "Asks user if they want to archive changes to remote repository after each Q&A session. Invoke at the end of each interaction to prompt for git commit and push."
---

# Git Archive Skill

## Purpose

This skill helps users manage version control for the vLLM-Ascend learning project by asking if they want to archive changes to the remote repository after each Q&A session.

## When to Use

Invoke this skill at the end of each interaction when:
- A user has completed a learning session
- Changes have been made to learning materials
- New content has been added or updated
- The user might want to save their progress

## Usage

After completing a Q&A session, the skill will:
1. Check for any uncommitted changes
2. List the modified files
3. Ask if the user wants to archive these changes
4. If yes, prompt for a commit message
5. Commit the changes and push to remote

## Example Workflow

1. User asks about vLLM-Ascend architecture
2. Assistant provides detailed explanation
3. Assistant asks if user wants to add this explanation to learning materials
4. User agrees, and new content is added to Week 1 documentation
5. Assistant invokes git-archive skill
6. Skill checks for changes and lists modified files
7. Skill asks: "Would you like to archive these changes to the remote repository?"
8. User responds "yes"
9. Skill asks for commit message
10. User provides commit message: "Add architecture explanation to Week 1"
11. Skill commits changes and pushes to remote

## Git Operations

The skill performs the following git operations:
- `git status` - Check for changes
- `git add .` - Stage all changes
- `git commit -m "<message>"` - Commit with user-provided message
- `git push` - Push to remote repository

## Configuration

The skill assumes the repository is already initialized and connected to a remote. If not, it will provide instructions for setting up git remote.

## Best Practices

- Use descriptive commit messages
- Archive changes regularly to avoid losing work
- Pull before pushing if working with a team
- Review changes before committing

## Troubleshooting

If git operations fail, the skill will:
1. Provide error message
2. Suggest possible fixes
3. Allow user to retry or skip archiving

## Contributing

To improve this skill:
1. Fork the repository
2. Update the SKILL.md file
3. Submit a pull request

---

**Note:** This skill is designed specifically for the vLLM-Ascend learning project but can be adapted for other git-managed projects.