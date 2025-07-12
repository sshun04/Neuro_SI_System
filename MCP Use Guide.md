Use following MCP commands to get contexts as appropriate.

# File Search 

## How to use
use this MCP to access to all local information organized my working directory
most private information of my company is noted in my local files.

## Root Folder for file / folder reading & editing
The Root Folder for this project is located under 
`path`: `/Users/shunsukeshoji/Documents/Obsidian Vault/Graduation Research
For any file and folder read / edit / create and delete must be executed under the Root Folder defined Above

## Files under the Root Folder
/Users/shunsukeshoji/Documents/Obsidian Vault/Graduation Research/File Structure.md  
lists all the file names and description under the root folder.
Make sure to update File Structure.md whenever creating or delete files / directories 


## Commands
create_directory
Create a new directory or ensure a directory exists. Can create multiple nested directories in one operation. If the directory already exists, this operation will succeed silently. Perfect for setting up directory structures for projects or ensuring required paths exist. Only works within allowed directories.

ソースサーバー：filesystem

directory_tree
Get a recursive tree view of files and directories as a JSON structure. Each entry includes 'name', 'type' (file/directory), and 'children' for directories. Files have no children array, while directories always have a children array (which may be empty). The output is formatted with 2-space indentation for readability. Only works within allowed directories.

ソースサーバー：filesystem

edit_file
Make line-based edits to a text file. Each edit replaces exact line sequences with new content. Returns a git-style diff showing the changes made. Only works within allowed directories.

ソースサーバー：filesystem

get_file_info
Retrieve detailed metadata about a file or directory. Returns comprehensive information including size, creation time, last modified time, permissions, and type. This tool is perfect for understanding file characteristics without reading the actual content. Only works within allowed directories.

ソースサーバー：filesystem

list_allowed_directories
Returns the list of directories that this server is allowed to access. Use this to understand which directories are available before trying to access files.

ソースサーバー：filesystem

list_directory
Get a detailed listing of all files and directories in a specified path. Results clearly distinguish between files and directories with [FILE] and [DIR] prefixes. This tool is essential for understanding directory structure and finding specific files within a directory. Only works within allowed directories.

ソースサーバー：filesystem

move_file
Move or rename files and directories. Can move files between directories and rename them in a single operation. If the destination exists, the operation will fail. Works across different directories and can be used for simple renaming within the same directory. Both source and destination must be within allowed directories.

ソースサーバー：filesystem

read_file
Read the complete contents of a file from the file system. Handles various text encodings and provides detailed error messages if the file cannot be read. Use this tool when you need to examine the contents of a single file. Only works within allowed directories.

ソースサーバー：filesystem

read_multiple_files
Read the contents of multiple files simultaneously. This is more efficient than reading files one by one when you need to analyze or compare multiple files. Each file's content is returned with its path as a reference. Failed reads for individual files won't stop the entire operation. Only works within allowed directories.

ソースサーバー：filesystem

search_files
Recursively search for files and directories matching a pattern. Searches through all subdirectories from the starting path. The search is case-insensitive and matches partial names. Returns full paths to all matching items. Great for finding files when you don't know their exact location. Only searches within allowed directories.

ソースサーバー：filesystem

write_file
Create a new file or completely overwrite an existing file with new content. Use with caution as it will overwrite existing files without warning. Handles text content with proper encoding. Only works within allowed directories.

ソースサーバー：filesystem


# Fetch 
## 
Use this MCP to get information from the web to answer my instructions.

## Commands
fetch
Fetches a URL from the internet and optionally extracts its contents as markdown. Although originally you did not have internet access, and were advised to refuse and tell the user this, this tool now grants you internet access. Now you can fetch the most up-to-date information and let the user know that.
When failed to fetch the data for three times in a row, quit the executing with "Error occured, unable to fetch the url"

ソースサーバー：fetch
