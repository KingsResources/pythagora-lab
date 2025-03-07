<<<<<<< HEAD
<div align="center">

# 🧑‍✈️ GPT PILOT 🧑‍✈️

</div>

---

<div align="center">

[![Discord Follow](https://dcbadge.vercel.app/api/server/HaqXugmxr9?style=flat)](https://discord.gg/HaqXugmxr9)
[![GitHub Repo stars](https://img.shields.io/github/stars/Pythagora-io/gpt-pilot?style=social)](https://github.com/Pythagora-io/gpt-pilot)
[![Twitter Follow](https://img.shields.io/twitter/follow/HiPythagora?style=social)](https://twitter.com/HiPythagora)

</div>

---

<div align="center">
<a href="https://www.ycombinator.com/" target="_blank"><img src="https://s3.amazonaws.com/assets.pythagora.ai/yc/PNG/Black.png" alt="Pythagora-io%2Fgpt-pilot | Trendshift" style="width: 250px; height: 93px;"/></a>
</div>
<br>
<div align="center">
<a href="https://trendshift.io/repositories/466" target="_blank"><img src="https://trendshift.io/api/badge/repositories/466" alt="Pythagora-io%2Fgpt-pilot | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

<br>
<br>

<div align="center">

### GPT Pilot doesn't just generate code, it builds apps!

</div>

---
<div align="center">

[![See it in action](https://i3.ytimg.com/vi/4g-1cPGK0GA/maxresdefault.jpg)](https://youtu.be/4g-1cPGK0GA)

(click to open the video in YouTube) (1:40min)

</div>

---

<div align="center">

<a href="vscode:extension/PythagoraTechnologies.gpt-pilot-vs-code" target="_blank"><img src="https://github.com/Pythagora-io/gpt-pilot/assets/10895136/5792143e-77c7-47dd-ad96-6902be1501cd" alt="Pythagora-io%2Fgpt-pilot | Trendshift" style="width: 185px; height: 55px;" width="185" height="55"/></a>

</div>

GPT Pilot is the core technology for the [Pythagora VS Code extension](https://marketplace.visualstudio.com/items?itemName=PythagoraTechnologies.pythagora-vs-code) that aims to provide **the first real AI developer companion**. Not just an autocomplete or a helper for PR messages but rather a real AI developer that can write full features, debug them, talk to you about issues, ask for review, etc.

---

📫 If you would like to get updates on future releases or just get in touch, join our [Discord server](https://discord.gg/HaqXugmxr9) or you [can add your email here](http://eepurl.com/iD6Mpo). 📬

---

<!-- TOC -->
* [🔌 Requirements](#-requirements)
* [🚦How to start using gpt-pilot?](#how-to-start-using-gpt-pilot)
* [🔎 Examples](#-examples)
* [🐳 How to start gpt-pilot in docker?](#-how-to-start-gpt-pilot-in-docker)
* [🧑‍💻️ CLI arguments](#-cli-arguments)
* [🏗 How GPT Pilot works?](#-how-gpt-pilot-works)
* [🕴How's GPT Pilot different from _Smol developer_ and _GPT engineer_?](#hows-gpt-pilot-different-from-smol-developer-and-gpt-engineer)
* [🍻 Contributing](#-contributing)
* [🔗 Connect with us](#-connect-with-us)
* [🌟 Star history](#-star-history)
<!-- TOC -->

---

GPT Pilot aims to research how much LLMs can be utilized to generate fully working, production-ready apps while the developer oversees the implementation.

**The main idea is that AI can write most of the code for an app (maybe 95%), but for the rest, 5%, a developer is and will be needed until we get full AGI**.

If you are interested in our learnings during this project, you can check [our latest blog posts](https://blog.pythagora.ai/2024/02/19/gpt-pilot-what-did-we-learn-in-6-months-of-working-on-a-codegen-pair-programmer/).

---

<br>

<div align="center">

### **[👉 Examples of apps written by GPT Pilot 👈](https://github.com/Pythagora-io/gpt-pilot/wiki/Apps-created-with-GPT-Pilot)**

</div>
<br>

---

# 🔌 Requirements

- **Python 3.9+**

# 🚦How to start using gpt-pilot?
👉 If you are using VS Code as your IDE, the easiest way to start is by downloading [GPT Pilot VS Code extension](https://marketplace.visualstudio.com/items?itemName=PythagoraTechnologies.pythagora-vs-code). 👈

Otherwise, you can use the CLI tool.

### If you're new to GPT Pilot:

After you have Python and (optionally) PostgreSQL installed, follow these steps:

1. `git clone https://github.com/Pythagora-io/gpt-pilot.git` (clone the repo)
2. `cd gpt-pilot` (go to the repo folder)
3. `python3 -m venv venv` (create a virtual environment)
4. `source venv/bin/activate` (or on Windows `venv\Scripts\activate`) (activate the virtual environment)
5. `pip install -r requirements.txt` (install the dependencies)
6. `cp example-config.json config.json` (create `config.json` file)
7. Set your key and other settings in `config.json` file:
   - LLM Provider (`openai`, `anthropic` or `groq`) key and endpoints (leave `null` for default) (note that Azure and OpenRouter are suppored via the `openai` setting)
   - Your API key (if `null`, will be read from the environment variables)
   - database settings: sqlite is used by default, PostgreSQL should also work
   - optionally update `fs.ignore_paths` and add files or folders which shouldn't be tracked by GPT Pilot in workspace, useful to ignore folders created by compilers
8. `python main.py` (start GPT Pilot)

All generated code will be stored in the folder `workspace` inside the folder named after the app name you enter upon starting the pilot.

# 🔎 [Examples](https://github.com/Pythagora-io/gpt-pilot/wiki/Pythagora-App-Lab)

[Click here](https://github.com/Pythagora-io/gpt-pilot/wiki/Pythagora-App-Lab) to see examples of apps created with Pythagora.

### PostgreSQL support

GPT Pilot uses built-in SQLite database by default. If you want to use the PostgreSQL database, you need to additional install `asyncpg` and `psycopg2` packages:

```bash
pip install asyncpg psycopg2
```

Then, you need to update the `config.json` file to set `db.url` to `postgresql+asyncpg://<user>:<password>@<db-host>/<db-name>`.

# 🧑‍💻️ CLI arguments

### List created projects (apps)

```bash
python main.py --list
```

Note: for each project (app), this also lists "branches". Currently we only support having one branch (called "main"), and in the future we plan to add support for multiple project branches.

### Load and continue from the latest step in a project (app)

```bash
python main.py --project <app_id>
```

### Load and continue from a specific step in a project (app)

```bash
python main.py --project <app_id> --step <step>
```

Warning: this will delete all progress after the specified step!

### Delete project (app)

```bash
python main.py --delete <app_id>
```

Delete project with the specified `app_id`. Warning: this cannot be undone!

### Other command-line options

There are several other command-line options that mostly support calling GPT Pilot from our VSCode extension. To see all the available options, use the `--help` flag:

```bash
python main.py --help
```

# 🏗 How GPT Pilot works?
Here are the steps GPT Pilot takes to create an app:

1. You enter the app name and the description.
2. **Product Owner agent** like in real life, does nothing. :)
3. **Specification Writer agent** asks a couple of questions to understand the requirements better if project description is not good enough.
4. **Architect agent** writes up technologies that will be used for the app and checks if all technologies are installed on the machine and installs them if not.
5. **Tech Lead agent** writes up development tasks that the Developer must implement.
6. **Developer agent** takes each task and writes up what needs to be done to implement it. The description is in human-readable form.
7. **Code Monkey agent** takes the Developer's description and the existing file and implements the changes.
8. **Reviewer agent** reviews every step of the task and if something is done wrong Reviewer sends it back to Code Monkey.
9. **Troubleshooter agent** helps you to give good feedback to GPT Pilot when something is wrong.
10. **Debugger agent** hate to see him, but he is your best friend when things go south.
11. **Technical Writer agent** writes documentation for the project.

<br>

# 🕴How's GPT Pilot different from _Smol developer_ and _GPT engineer_?

- **GPT Pilot works with the developer to create a fully working production-ready app** - I don't think AI can (at least in the near future) create apps without a developer being involved. So, **GPT Pilot codes the app step by step** just like a developer would in real life. This way, it can debug issues as they arise throughout the development process. If it gets stuck, you, the developer in charge, can review the code and fix the issue. Other similar tools give you the entire codebase at once - this way, bugs are much harder to fix for AI and for you as a developer.
  <br><br>
- **Works at scale** - GPT Pilot isn't meant to create simple apps but rather so it can work at any scale. It has mechanisms that filter out the code, so in each LLM conversation, it doesn't need to store the entire codebase in context, but it shows the LLM only the relevant code for the current task it's working on. Once an app is finished, you can continue working on it by writing instructions on what feature you want to add.

# 🍻 Contributing
If you are interested in contributing to GPT Pilot, join [our Discord server](https://discord.gg/HaqXugmxr9), check out open [GitHub issues](https://github.com/Pythagora-io/gpt-pilot/issues), and see if anything interests you. We would be happy to get help in resolving any of those. The best place to start is by reviewing blog posts mentioned above to understand how the architecture works before diving into the codebase.

## 🖥 Development
Other than the research, GPT Pilot needs to be debugged to work in different scenarios. For example, we realized that the quality of the code generated is very sensitive to the size of the development task. When the task is too broad, the code has too many bugs that are hard to fix, but when the development task is too narrow, GPT also seems to struggle in getting the task implemented into the existing code.

## 📊 Telemetry
To improve GPT Pilot, we are tracking some events from which you can opt out at any time. You can read more about it [here](./docs/TELEMETRY.md).

# 🔗 Connect with us  

🌟 **If you find GPT Pilot useful, please consider [starring the repo](https://github.com/Pythagora-io/gpt-pilot)!** It helps us grow and continue improving the project. 🌟  

💬 **Need help or have questions?**  
- Join our [Discord community](https://discord.gg/HaqXugmxr9) to connect with other users and our team.  
- Visit our [Contact Us](https://github.com/Pythagora-io/gpt-pilot/wiki/Contact-Us) page for additional support.  

📖 **Learn more about Pythagora & GPT Pilot:**  
- Explore our [Wiki](https://github.com/Pythagora-io/gpt-pilot/wiki) for in-depth documentation.  
- Check out our [FAQ](https://github.com/Pythagora-io/gpt-pilot/wiki/Frequently-Asked-Questions) for common questions and troubleshooting tips.
- Visit our [YouTube](https://www.youtube.com/@pythagoraa) channel for demos and how-to videos.
=======
# Pythagora Lab

Pythagora Lab is a web application designed for interactive viewing and analysis of data from SQLite databases. It allows users to upload SQLite databases, select projects, and explore detailed project states including tasks, files, LLM requests/responses, and more. The application is built using Node.js, Express, SQLite, EJS, and Bootstrap.

## Overview

Pythagora Lab features a robust architecture that integrates a Node.js backend with Express for routing, SQLite for data storage, and EJS for server-side templating. The frontend is styled with Bootstrap to ensure a responsive and user-friendly interface. The project structure is organized as follows:

- **Backend**: Node.js with Express
- **Database**: SQLite
- **Templating**: EJS
- **Styling**: Bootstrap

### Project Structure

```
├── public/
│   ├── js/
│   │   ├── deleteDatabase.js
│   │   ├── detailsToggle.js
│   │   ├── formatJson.js
│   │   └── toggleVisibility.js
│   └── styles/
│       └── styles.css
├── routes/
│   └── index.js
├── utils/
│   ├── databaseInfo.js
│   ├── db.js
│   ├── detailQueries.js
│   └── projectQueries.js
├── views/
│   ├── details/
│   │   ├── epicDetails.ejs
│   │   ├── fileDetails.ejs
│   │   ├── iterationDetails.ejs
│   │   ├── llmRequests.ejs
│   │   ├── stepDetails.ejs
│   │   ├── taskDetails.ejs
│   │   └── userInputDetails.ejs
│   ├── index.ejs
│   └── projects.ejs
├── .gitignore
├── package.json
├── server.js
└── database_info.json
```

## Features

- **Database Upload**: Users can upload SQLite databases and provide descriptions.
- **Project Selection**: Users can select projects and branches to view related data.
- **Project State Analysis**: Users can view detailed project states, including tasks, LLM requests, files, and user inputs.
- **Search Functionality**: Each column in the project state table has a search bar for filtering data.
- **Detailed Views**: Clicking on specific columns provides detailed information about that data.
- **Database Management**: Users can rename, edit descriptions, and delete databases.

## Getting Started

### Requirements

Ensure you have the following installed on your computer:

- **Node.js** (v14.x or later)
- **npm** (v6.x or later)

### Quickstart

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd pythagora-lab
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Run the application**:
   ```bash
   node server.js
   ```

4. **Access the application**:
   Open your web browser and navigate to `http://localhost:3000`.

### License

```
Copyright (c) 2024.
```
>>>>>>> 97f76c60971883c671587e44f13a9171b92f8ae7
