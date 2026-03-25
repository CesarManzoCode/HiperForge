"""
Tests unitarios para Project y Workspace — ciclo de vida, tags, jerarquía.
"""

from __future__ import annotations

import pytest

from hiperforge.domain.entities.project import Project, ProjectStatus
from hiperforge.domain.entities.workspace import Workspace, WorkspaceStatus
from hiperforge.domain.entities.task import Task
from hiperforge.domain.exceptions import InvalidStatusTransition


# ═══════════════════════════════════════════════════════════════════
# PROJECT
# ═══════════════════════════════════════════════════════════════════

class TestProjectCreation:

    def test_create_genera_id_unico(self):
        p1 = Project.create(name="A", workspace_id="ws1")
        p2 = Project.create(name="B", workspace_id="ws1")
        assert p1.id != p2.id

    def test_create_estado_active(self):
        p = Project.create(name="Test", workspace_id="ws1")
        assert p.status == ProjectStatus.ACTIVE

    def test_create_preserva_nombre_y_descripcion(self):
        p = Project.create(name="API", workspace_id="ws1", description="Backend")
        assert p.name == "API"
        assert p.description == "Backend"

    def test_create_sin_tags(self):
        p = Project.create(name="Test", workspace_id="ws1")
        assert len(p.tags) == 0

    def test_create_con_tags(self):
        p = Project.create(name="Test", workspace_id="ws1", tags=["python", "api"])
        assert "python" in p.tags
        assert "api" in p.tags


class TestProjectTransitions:

    def test_active_a_archived(self, sample_project):
        archived = sample_project.archive()
        assert archived.status == ProjectStatus.ARCHIVED

    def test_archived_a_active(self, sample_project):
        archived = sample_project.archive()
        reactivated = archived.reactivate()
        assert reactivated.status == ProjectStatus.ACTIVE

    def test_active_a_deleted(self, sample_project):
        deleted = sample_project.delete()
        assert deleted.status == ProjectStatus.DELETED

    def test_deleted_es_terminal(self, sample_project):
        deleted = sample_project.delete()
        with pytest.raises(InvalidStatusTransition):
            deleted.archive()

    def test_add_task_a_proyecto(self, sample_project):
        task = Task.create(prompt="test", project_id=sample_project.id)
        updated = sample_project.add_task(task)
        assert len(updated.tasks) == 1

    def test_rename_proyecto(self, sample_project):
        renamed = sample_project.rename("Nuevo nombre")
        assert renamed.name == "Nuevo nombre"


class TestProjectTags:

    def test_add_tag(self, sample_project):
        updated = sample_project.add_tag("fastapi")
        assert "fastapi" in updated.tags

    def test_remove_tag(self, sample_project):
        with_tag = sample_project.add_tag("temp")
        without_tag = with_tag.remove_tag("temp")
        assert "temp" not in without_tag.tags

    def test_add_tag_duplicado_no_duplica(self, sample_project):
        p = sample_project.add_tag("python")
        p = p.add_tag("python")
        assert p.tags.count("python") == 1


# ═══════════════════════════════════════════════════════════════════
# WORKSPACE
# ═══════════════════════════════════════════════════════════════════

class TestWorkspaceCreation:

    def test_create_genera_id_unico(self):
        w1 = Workspace.create(name="A")
        w2 = Workspace.create(name="B")
        assert w1.id != w2.id

    def test_create_estado_active(self):
        ws = Workspace.create(name="test")
        assert ws.status == WorkspaceStatus.ACTIVE

    def test_create_preserva_nombre(self):
        ws = Workspace.create(name="mi-workspace", description="desc")
        assert ws.name == "mi-workspace"
        assert ws.description == "desc"

    def test_create_schema_version_actual(self):
        ws = Workspace.create(name="test")
        assert ws.schema_version == Workspace.CURRENT_SCHEMA_VERSION

    def test_create_sin_proyectos(self):
        ws = Workspace.create(name="test")
        assert len(ws.projects) == 0


class TestWorkspaceTransitions:

    def test_active_a_archived(self, sample_workspace):
        archived = sample_workspace.archive()
        assert archived.status == WorkspaceStatus.ARCHIVED

    def test_archived_a_active(self, sample_workspace):
        archived = sample_workspace.archive()
        reactivated = archived.reactivate()
        assert reactivated.status == WorkspaceStatus.ACTIVE

    def test_deleted_es_terminal(self, sample_workspace):
        deleted = sample_workspace.delete()
        with pytest.raises(InvalidStatusTransition):
            deleted.archive()

    def test_rename_workspace(self, sample_workspace):
        renamed = sample_workspace.rename("nuevo")
        assert renamed.name == "nuevo"

    def test_add_project(self, sample_workspace, sample_project):
        updated = sample_workspace.add_project(sample_project)
        assert len(updated.projects) == 1
