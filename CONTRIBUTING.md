# Contributing to EP-BOLFI

If you'd like to contribute to EP-BOLFI, thank you very much and please have a look at the guidelines below.

## Workflow

We use [GIT](https://en.wikipedia.org/wiki/Git) and [GitLab](https://en.wikipedia.org/wiki/GitLab) to coordinate our work. When making any kind of update, we try to follow the procedure below.

### Before you begin

1. Create an [issue](https://github.com/features/issues) where new proposals can be discussed before any coding is done.
2. Download the source code onto your local system, by cloning the repository:
```bash
git clone https://github.com/YannickNoelStephanKuhn/EP-BOLFI
```
3. Install the library in editable mode:
```bash
pip install -e ep-bolfi
```
4. Create a branch of this repo, where all changes will be made, and "checkout" that branch so that your changes live in that branch:
```bash
git branch <branch_name>
git checkout <branch_name>
```
Or as a short-hand:
```bash
git checkout -b <branch_name>
```

### Writing your code

4. EP-BOLFI is written in [Python](https://en.wikipedia.org/wiki/Python_(programming_language)), and makes heavy use of [ELFI](https://github.com/elfi-dev/elfi) as well as [PyBaMM](https://github.com/pybamm-team/PyBaMM).
5. Make sure to follow our [coding style guidelines](#coding-style-guidelines).
6. Commit your changes to your branch with [useful, descriptive commit messages](https://chris.beams.io/posts/git-commit/): Remember these are visible to all and should still make sense a few months ahead in time. While developing, you can keep using the GitLab issue you're working on as a place for discussion. [Refer to your commits](https://stackoverflow.com/questions/8910271/how-can-i-reference-a-commit-in-an-issue-comment-on-github) when discussing specific lines of code.
7. If you want to add a dependency on another library, or re-use code you found somewhere else, have a look at [these guidelines](#dependencies-and-reusing-code).

### Merging your changes with EP-BOLFI

8. Make sure that your code runs successfully. Ideally, implement tests.
9. Run `flake8` on your code to fix formatting issues ahead of time.
10. When you feel your code is finished, or at least warrants serious discussion, create a [merge request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/incorporating-changes-from-a-pull-request/merging-a-pull-request) (MR) on the [GitHub page of EP-BOLFI](https://github.com/YannickNoelStephanKuhn/EP-BOLFI).
11. Once a MR has been created, it will be reviewed by any member of the group. Changes might be suggested which you can make by simply adding new commits to the branch. When everything's finished, someone with the right GitLab permissions will merge your changes into the EP-BOLFI main repository.

## Coding style guidelines

EP-BOLFI follows the [PEP8 recommendations](https://www.python.org/dev/peps/pep-0008/) for coding style. These are very common guidelines, and community tools have been developed to check how well projects implement them.

### Flake8

We use [flake8](http://flake8.pycqa.org/en/latest/) to check our PEP8 adherence. To try this on your system, navigate to the ep_bolfi directory in a terminal and type:

```bash
flake8 --extend-ignore E266
```

We ignore the double hashes (E266), since we need them for member documentation with Doxygen. The .flake8 file does this automatically, but only in the top folder.

### Documentation

The documentation is generated with [Doxygen](https://www.doxygen.nl/) from the source code.

Hence, please copy the structure of the in-code documentation for your own comments. Here are some less obvious technicalities:
 - Member documentation is supposed to work with something but ##, but it doesn't. Hence please use ## for comments, or they will not show up tin the documentation.
 - Empty lines in docstrings (the """ ... """ ones) make Doxygen treat everything after them as plain text. So do not put empty lines in docstrings.

## Dependencies and reusing code

While it's a bad idea for developers to "reinvent the wheel", it's important for users to get a _reasonably_ sized download and an easy install. In addition, external libraries can sometimes cease to be supported, and when they contain bugs it might take a while before fixes become available as automatic downloads to EP-BOLFI users.
For these reasons, all dependencies in EP-BOLFI should be thought about carefully, and discussed on GitHub.

Direct inclusion of code from other packages is possible, as long as their license permits it and is compatible with ours, but again should be considered carefully and discussed in the group. Snippets from blogs and [stackoverflow](https://stackoverflow.com/) are often incompatible with other licences than [CC BY-SA](https://creativecommons.org/licenses/by-sa/4.0/) and hence should be avoided. You should attribute (and document) any included code from other packages, by making a comment with a link in the source code.

## Building from source

### Build and install wheel from source (pip)

Install the build command and execute it:
```bash
pip install build
python3 -m build
```

The wheel file should be at dist/ep_bolfi-${VERSION}-py3-none-any.whl. Please do not commit these.

## Infrastructure

### GitLab

GitLab does some magic with particular filenames. In particular:

- The first page people see when they go to [our GitHub page](https://github.com/YannickNoelStephanKuhn/EP-BOLFI) displays the contents of [README.md](README.md), which is written in the [Markdown](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet) format. Some guidelines can be found [here](https://help.github.com/articles/about-readmes/).
- This file, [CONTRIBUTING.md](CONTRIBUTING.md) is recognised as the contribution guidelines and a link is [automatically](https://github.com/blog/1184-contributing-guidelines) displayed when new issues or pull requests are created.

## Acknowledgements

This CONTRIBUTING.md file was adapted from the excellent [Pints GitHub repo](https://github.com/pints-team/pints).
