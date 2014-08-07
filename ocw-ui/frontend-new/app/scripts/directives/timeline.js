/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *    http: *www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

'use strict';

/**
 * @ngdoc directive
 * @name ocwUiApp.directive:timeline
 * @description
 * # timeline
 */
angular.module('ocwUiApp')
.directive('timeline', function($rootScope, $window) {
	return {
		restrict: 'C',
		replace: true,
		transclude: true,
		template: '<div id="OCWtimeline"></div>',
		link: function(scope, element, attrs) {
			// Instantiate timeline object.
			$rootScope.timeline = new links.Timeline(document.getElementById('OCWtimeline'));

			// Redraw the timeline whenever the window is resized
			angular.element($window).bind('resize', function() {
				$rootScope.timeline.checkResize();
			});

			var options = {
				"width": "100%",
				"showCurrentTime": false,
				"moveable": false,
				"zoomable": false
			};

			$rootScope.timeline.draw([], options);
		}
	}
});
