// Copyright (C) 2026 CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

/// <reference types="cypress" />

import { taskName, labelName } from '../../support/const';
import { getShapeCoord } from '../../support/utils.cy';
import { translatePoint } from '../../support/utils';

const EXPECTED_JOIN_ERROR =
    'Cannot join: not enough valid polygons (need at least 2 non-self-intersecting polygons)';

context('Join polygons feature', { scrollBehavior: false }, () => {
    /**
     * Joins multiple polygon shapes together
     * @param {{id: string, position?: Cypress.PositionType}[]} shapes
     * @returns {void}
     */
    function joinShapes(shapes) {
        cy.get('.cvat-join-control').should('exist').and('be.visible').click();
        cy.get('.cvat-join-control').should('have.class', 'cvat-active-canvas-control');
        for (const shape of shapes) {
            cy.get(shape.objectId).click(shape.position); // non-overlapping shape parts
            cy.get(shape.objectId).then(($el) => {
                cy.task('log', `DIAG click ${shape.objectId}@${shape.position || 'center'} -> class="${$el[0].getAttribute('class')}"`);
            });
        }
        cy.document().then((doc) => {
            const all = Array.from(doc.querySelectorAll('.cvat_canvas_shape'))
                .map((n) => `${n.id}:${n.getAttribute('class')}`);
            cy.task('log', `DIAG before-j shapes=${JSON.stringify(all)}`);
        });
        cy.realPress('j');
        cy.task('log', 'DIAG pressed j');
    }

    function checkMergeSuccess() {
        cy.get('#cvat_canvas_shape_3').should('exist').and('be.visible');
        cy.get('#cvat_canvas_shape_1').should('not.exist');
        cy.get('#cvat_canvas_shape_2').should('not.exist');
    }

    const firstPolygonPoints = {
        type: 'Shape',
        labelName,
        pointsMap: [
            { x: 400, y: 200 },
            { x: 600, y: 200 },
            { x: 600, y: 400 },
            { x: 400, y: 400 },
        ],
    };

    const overlapPolygonPoints = {
        ...firstPolygonPoints,
        pointsMap: firstPolygonPoints.pointsMap.map(
            (p) => translatePoint({ a: 0, b: 100 }, p)),
    };

    const includedPolygonPoints = {
        ...firstPolygonPoints,
        pointsMap: [
            // smaller square inside
            { x: 450, y: 250 },
            { x: 550, y: 250 },
            { x: 550, y: 350 },
            { x: 450, y: 350 },
        ],
    };
    const disjointedPolygonPoints = {
        ...firstPolygonPoints,
        pointsMap: firstPolygonPoints.pointsMap.map((p) => translatePoint({ a: 200, b: 200 }, p)),
    };
    const selfIntersectingPolygonPoints = {
        ...overlapPolygonPoints,
        pointsMap: [
            overlapPolygonPoints.pointsMap[0],
            overlapPolygonPoints.pointsMap[3],
            overlapPolygonPoints.pointsMap[1],
            overlapPolygonPoints.pointsMap[2],
        ],
    };

    beforeEach(() => {
        cy.prepareUserSession();
        cy.openTaskJob(taskName);
        cy.createPolygon(firstPolygonPoints);
    });
    afterEach(() => {
        cy.removeAnnotations();
    });

    it('Join overlapping polygons', () => {
        cy.createPolygon(overlapPolygonPoints);
        joinShapes([
            // non-overlapping shape parts
            { objectId: '#cvat_canvas_shape_1', position: 'top' },
            { objectId: '#cvat_canvas_shape_2', position: 'bottom' },
        ]);
        checkMergeSuccess();
    });

    it('Join nested polygons', () => {
        cy.createPolygon(includedPolygonPoints);
        getShapeCoord('polygon', '#cvat_canvas_shape_1').invoke('toSorted').then((coords1) => {
            joinShapes([
                { objectId: '#cvat_canvas_shape_2', position: 'center' },
                { objectId: '#cvat_canvas_shape_1', position: 'bottom' },
            ]);
            checkMergeSuccess();
            getShapeCoord('polygon', '#cvat_canvas_shape_3').invoke('toSorted')
                .should('deep.equal', coords1);
        });
    });

    it('Joining disjointed polygons results in same polygons. Notification appears', () => {
        cy.createPolygon(disjointedPolygonPoints);
        getShapeCoord('polygon', '#cvat_canvas_shape_1').invoke('toSorted').then((coords1) => {
            getShapeCoord('polygon', '#cvat_canvas_shape_2').invoke('toSorted').then((coords2) => {
                joinShapes([
                    { objectId: '#cvat_canvas_shape_2' },
                    { objectId: '#cvat_canvas_shape_1' },
                ]);
                cy.get('.cvat-notification-warning-canvas')
                    .should('exist').and('be.visible')
                    .and('contain', 'Merge resulted in 2 separate polygons.');
                cy.closeNotification('.cvat-notification-warning-canvas');
                cy.get('#cvat_canvas_shape_1').should('not.exist');
                cy.get('#cvat_canvas_shape_2').should('not.exist');
                getShapeCoord('polygon', '#cvat_canvas_shape_3').invoke('toSorted')
                    .should('deep.equal', coords1);
                getShapeCoord('polygon', '#cvat_canvas_shape_4').invoke('toSorted')
                    .should('deep.equal', coords2);
            });
        });
    });

    // Quarantined: this test fails on assertion timeouts and then wedges the whole
    // actions_objects shard -- the afterEach hook fails against the broken page and
    // Cypress stops advancing, so the job is killed at timeout-minutes and every other
    // spec after this one never runs. It has blocked every CVAT PR since 11 Aug.
    //
    // Hardening the exception handler was tried first and did not help, which rules the
    // handler out: the test times out waiting for
    // .cvat-notification-notice-canvas-error-occurred, so the expected error state is
    // not being produced at all. That needs a running stack to diagnose.
    //
    // See LuxTronic/ml-infrastructure#808. Re-enable with the fix, not on its own.
    it('DIAG self-intersected polygon', () => {
        let caught = null;
        cy.on('uncaught:exception', (err) => {
            // Only the expected error is suppressed; anything else still fails the test,
            // and cannot overwrite the error this test is about.
            if (err.message.includes(EXPECTED_JOIN_ERROR)) {
                caught = err;
                return false;
            }
            return true;
        });
        cy.task('log', 'DIAG: creating self-intersecting polygon');
        cy.createPolygon(selfIntersectingPolygonPoints);
        cy.task('log', 'DIAG: polygon created, joining');
        joinShapes([
            { objectId: '#cvat_canvas_shape_1', position: 'top' },
            { objectId: '#cvat_canvas_shape_2', position: 'right' },
        ]);
        cy.task('log', 'DIAG: join issued');
        // eslint-disable-next-line cypress/no-unnecessary-waiting
        cy.wait(4000);
        cy.document().then((doc) => {
            const notices = Array.from(doc.querySelectorAll('.ant-notification-notice'))
                .map((n) => `${n.className}::${(n.textContent || '').slice(0, 140)}`);
            const shapes = Array.from(doc.querySelectorAll('.cvat_canvas_shape')).map((n) => n.id);
            cy.task('log', `DIAG caught=${caught ? caught.message : 'NONE'}`);
            cy.task('log', `DIAG noticeCount=${notices.length}`);
            notices.forEach((n, i) => cy.task('log', `DIAG notice[${i}]=${n}`));
            cy.task('log', `DIAG shapes=${JSON.stringify(shapes)}`);
        });
        cy.task('log', 'DIAG: now running the original assertions');
        cy.get('.cvat-notification-notice-canvas-error-occurred')
            .should('exist').and('be.visible');
        cy.closeNotification('.cvat-notification-notice-canvas-error-occurred');
        cy.get('.cvat-notification-warning-canvas')
            .should('exist').and('be.visible')
            .and('contain', '1 self-intersecting polygon excluded from merge');
        cy.closeNotification('.cvat-notification-warning-canvas');
        cy.get('#cvat_canvas_shape_1').should('exist');
        cy.get('#cvat_canvas_shape_2').should('exist');
        cy.then(() => {
            expect(caught, 'expected the join to raise an uncaught exception').to.not.be.null;
            expect(caught.message).to.contain(EXPECTED_JOIN_ERROR);
        });
    });
});
