# todo_app.py
"""
Todo App - Streamlit UI for task management.

A web-based interface for the mcp_todo task management system.
Run with: streamlit run todo_app.py
"""
import streamlit as st

from mcp_todo.storage import TodoStorage
from mcp_todo.models import TodoPriority, TodoStatus


def get_status_icon(status: TodoStatus) -> str:
    """Get display icon for status."""
    icons = {
        TodoStatus.PENDING: "[ ]",
        TodoStatus.IN_PROGRESS: "[~]",
        TodoStatus.COMPLETED: "[x]",
        TodoStatus.BLOCKED: "[!]",
        TodoStatus.CANCELLED: "[-]",
    }
    return icons.get(status, "[ ]")


def get_priority_color(priority: TodoPriority) -> str:
    """Get display color for priority."""
    colors = {
        TodoPriority.LOW: "gray",
        TodoPriority.MEDIUM: "blue",
        TodoPriority.HIGH: "orange",
        TodoPriority.URGENT: "red",
    }
    return colors.get(priority, "gray")


def main() -> None:
    st.set_page_config(
        page_title="Todo App",
        page_icon="todo",
        layout="wide",
    )

    st.title("Todo App")
    st.caption("Simple task management")

    # Initialize storage
    storage = TodoStorage()

    # Sidebar for adding new tasks
    with st.sidebar:
        st.header("Add New Task")

        new_title = st.text_input("Title", key="new_title")
        new_description = st.text_area("Description", key="new_desc", height=100)

        col1, col2 = st.columns(2)
        with col1:
            new_priority = st.selectbox(
                "Priority",
                options=["low", "medium", "high", "urgent"],
                index=1,
                key="new_priority",
            )
        with col2:
            new_project = st.text_input("Project", value="default", key="new_project")

        new_tags = st.text_input("Tags (comma-separated)", key="new_tags")

        if st.button("Add Task", type="primary", use_container_width=True):
            if new_title.strip():
                tags = [t.strip() for t in new_tags.split(",") if t.strip()]
                todo = storage.add(
                    title=new_title.strip(),
                    description=new_description.strip(),
                    priority=new_priority,
                    project=new_project.strip() or "default",
                    tags=tags,
                    ai_source="streamlit_ui",
                )
                st.success(f"Added: {todo.title} ({todo.id})")
                st.rerun()
            else:
                st.error("Title is required")

        st.divider()

        # Filters
        st.header("Filters")

        filter_status = st.selectbox(
            "Status",
            options=["all", "pending", "in_progress", "completed", "blocked", "cancelled"],
            key="filter_status",
        )

        filter_priority = st.selectbox(
            "Priority",
            options=["all", "low", "medium", "high", "urgent"],
            key="filter_priority",
        )

        hide_completed = st.checkbox("Hide completed", key="hide_completed")

    # Main content - Task list
    st.header("Tasks")

    # Get summary
    summary = storage.get_summary()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total", summary["total"])
    with col2:
        st.metric("Pending", summary["pending"])
    with col3:
        st.metric("In Progress", summary["in_progress"])
    with col4:
        st.metric("Completed", summary["completed"])

    st.divider()

    # Apply filters
    status_filter = None if filter_status == "all" else filter_status
    priority_filter = None if filter_priority == "all" else filter_priority

    todos = storage.list_all(
        status=status_filter,
        priority=priority_filter,
        include_completed=not hide_completed,
    )

    # Sort: in_progress first, then pending, then by priority
    priority_order = {"urgent": 0, "high": 1, "medium": 2, "low": 3}
    status_order = {"in_progress": 0, "pending": 1, "blocked": 2, "completed": 3, "cancelled": 4}

    todos.sort(key=lambda t: (
        status_order.get(t.status.value, 5),
        priority_order.get(t.priority.value, 4),
    ))

    if not todos:
        st.info("No tasks found. Add one using the sidebar!")
    else:
        for todo in todos:
            with st.container():
                col1, col2, col3 = st.columns([0.6, 0.25, 0.15])

                with col1:
                    # Display task info
                    status_icon = get_status_icon(todo.status)
                    priority_color = get_priority_color(todo.priority)

                    title_style = ""
                    if todo.status == TodoStatus.COMPLETED:
                        title_style = "~~"

                    st.markdown(
                        f"**{status_icon}** {title_style}{todo.title}{title_style} "
                        f":{priority_color}[{todo.priority.value}]"
                    )

                    if todo.description:
                        st.caption(todo.description)

                    if todo.tags:
                        st.caption(f"Tags: {', '.join(todo.tags)}")

                with col2:
                    # Status change buttons
                    btn_col1, btn_col2, btn_col3 = st.columns(3)

                    with btn_col1:
                        if todo.status != TodoStatus.IN_PROGRESS:
                            if st.button("Start", key=f"start_{todo.id}", use_container_width=True):
                                storage.start(todo.id, ai_source="streamlit_ui")
                                st.rerun()

                    with btn_col2:
                        if todo.status != TodoStatus.COMPLETED:
                            if st.button("Done", key=f"done_{todo.id}", use_container_width=True):
                                storage.complete(todo.id, ai_source="streamlit_ui")
                                st.rerun()

                    with btn_col3:
                        if todo.status == TodoStatus.COMPLETED:
                            if st.button("Reopen", key=f"reopen_{todo.id}", use_container_width=True):
                                storage.update(todo.id, status="pending", ai_source="streamlit_ui")
                                st.rerun()

                with col3:
                    # Delete button
                    if st.button("Delete", key=f"del_{todo.id}", type="secondary", use_container_width=True):
                        storage.delete(todo.id)
                        st.rerun()

                st.divider()

    # Footer actions
    st.header("Actions")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Clear Completed", use_container_width=True):
            count = storage.clear_completed()
            if count > 0:
                st.success(f"Cleared {count} completed tasks")
                st.rerun()
            else:
                st.info("No completed tasks to clear")

    with col2:
        if st.button("Export as Markdown", use_container_width=True):
            markdown = storage.export_markdown()
            st.download_button(
                label="Download Markdown",
                data=markdown,
                file_name="todos.md",
                mime="text/markdown",
            )

    with col3:
        if st.button("Refresh", use_container_width=True):
            st.rerun()


if __name__ == "__main__":
    main()
