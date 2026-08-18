// Copyright (C) 2021-2022 Intel Corporation
// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

/// <reference types="cypress" />

import { taskName } from '../../support/const';

context('Settings. "Auto save" option.', () => {
    const caseId = '51';

    // Seeds the persisted clientSettings blob through onBeforeLoad, so restoreSettingsAsync
    // reads it on boot the way it would in an annotator's browser. Seeding the live page and
    // reloading instead would race the running app, which rewrites clientSettings whenever
    // settings change, and would inherit whatever the previous test left behind.
    function seedStoredSettings(workspace, version) {
        const stored = { workspace };
        if (version !== null) {
            stored.version = version;
        }

        cy.url().then((url) => {
            cy.visit(url, {
                onBeforeLoad(win) {
                    win.localStorage.setItem('clientSettings', JSON.stringify(stored));
                },
            });
        });
        cy.get('.cvat-canvas-container').should('exist').and('be.visible');
    }

    // Read back from storage rather than the settings UI. updateCachedSettings writes the
    // resolved settings immediately after restoreSettingsAsync, so this is the value the
    // migration actually settled on -- asserted without depending on the header menu or the
    // modal, which have nothing to do with what is being tested here.
    function expectStoredAutoSave(expected) {
        cy.getAllLocalStorage().should((storage) => {
            const entry = Object.values(storage).find((origin) => origin.clientSettings);
            expect(entry, 'clientSettings present in local storage').to.not.equal(undefined);

            const stored = JSON.parse(entry.clientSettings);
            expect(stored.workspace.autoSave, 'persisted autoSave').to.equal(expected);
        });
    }

    before(() => {
        cy.prepareUserSession();
        cy.openTaskJob(taskName);
    });

    describe(`Testing case "${caseId}"`, () => {
        it('Auto save is enabled by default and shows the effective interval.', () => {
            cy.openSettings();
            cy.contains('Workspace').click();

            // Asserted before touching the control: this is the regression test for autosave
            // shipping enabled, so the checkbox must be observed in its default state.
            cy.get('.cvat-workspace-settings-auto-save').within(() => {
                cy.get('[type="checkbox"]').should('be.checked');
            });
            cy.get('.cvat-workspace-settings-auto-save-interval')
                .should('contain.text', '15 minutes');

            // Opting out still works, and is restored so the rest of the shard keeps the default.
            cy.get('.cvat-workspace-settings-auto-save').within(() => {
                cy.get('[type="checkbox"]').uncheck();
                cy.get('[type="checkbox"]').should('not.be.checked');
                cy.get('[type="checkbox"]').check();
                cy.get('[type="checkbox"]').should('be.checked');
            });
            cy.closeSettings();
        });

        it('Settings stored before versioning are migrated to autosave enabled.', () => {
            // An unversioned blob cannot distinguish the historical default false from a
            // deliberate opt-out, because nothing recorded the provenance of the choice.
            // The documented policy is to treat it as the old default and enable autosave;
            // this pins that policy so a future change to it has to be deliberate.
            seedStoredSettings({ autoSave: false }, null);

            expectStoredAutoSave(true);
        });

        it('An opt-out recorded after versioning survives the migration.', () => {
            // autoSavePreferenceSet is what the migration reads to tell a deliberate choice
            // apart from an inherited default, so an annotator who turns autosave off keeps
            // it off across reloads.
            seedStoredSettings({ autoSave: false, autoSavePreferenceSet: true }, 1);

            expectStoredAutoSave(false);

            // Leave the default in place for anything that runs after this spec.
            seedStoredSettings({ autoSave: true, autoSavePreferenceSet: true }, 1);
        });
    });
});
