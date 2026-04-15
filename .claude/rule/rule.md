
# Clean Code, SOLID & Reusability Ruleset

## 1. Clean Code Principles
* **Meaningful Names**: Use intention-revealing names for variables, functions, and classes. Avoid abbreviations (e.g., `user_repository` instead of `u_repo`).
* **Functions should do one thing**: Keep functions small (ideally < 20 lines). If a function performs multiple tasks, extract them into sub-functions.
* **DRY (Don't Repeat Yourself)**: Eliminate duplicate code. Abstract common logic into reusable utilities or base classes.
* **YAGNI (You Ain't Gonna Need It)**: Do not add functionality until it is deemed necessary. Avoid "just-in-case" code.
* **Dead Code Removal**: Regularly delete unused variables, functions, and commented-out code blocks.

## 2. SOLID Principles
* **Single Responsibility Principle (SRP)**: A class should have one, and only one, reason to change.
* **Open/Closed Principle (OCP)**: Software entities should be open for extension but closed for modification. Use interfaces or abstract classes.
* **Liskov Substitution Principle (LSP)**: Subtypes must be substitutable for their base types without altering correctness.
* **Interface Segregation Principle (ISP)**: Clients should not be forced to depend on methods they do not use. Prefer many small interfaces over one large one.
* **Dependency Inversion Principle (DIP)**: Depend on abstractions, not concretions. Use Dependency Injection (DI) to manage dependencies.

## 3. Reusability & Maintainability
* **Composition over Inheritance**: Prefer building complex logic by combining simpler objects rather than creating deep inheritance trees.
* **Pure Functions**: Where possible, write functions that return the same output for the same input and have no side effects.
* **Decoupling**: Minimize dependencies between modules. Use events or observers for cross-module communication.
* **Error Handling**: Use exceptions rather than return codes. Handle errors at the appropriate level.

## 4. Implementation Rules for Claude Code
* Always refactor code to meet these standards before finalizing.
* If a proposed change introduces redundancy, flag it and suggest an abstraction.
* Prioritize readability and long-term maintenance over clever "one-liners".
