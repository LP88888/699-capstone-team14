"""
Visualization stage: Generates PyVis network and Matplotlib charts.
"""
from __future__ import annotations

import html
import json
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pathlib import Path
from pyvis.network import Network
import shutil

from ..ingrnorm.dedupe_map import load_jsonl_map, _DROP_TOKENS, _DROP_SUBSTRINGS

from ..core import PipelineContext, StageResult
from ..utils import stage_logger


def _lighten(hex_color: str, factor: float = 0.35) -> str:
    """Lighten a hex color by blending toward white."""
    if not hex_color:
        return hex_color
    h = hex_color.lstrip("#")
    try:
        r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    except Exception:
        return hex_color
    r = min(255, int(r + (255 - r) * factor))
    g = min(255, int(g + (255 - g) * factor))
    b = min(255, int(b + (255 - b) * factor))
    return f"#{r:02x}{g:02x}{b:02x}"


def _blend_colors(c1: str, c2: str) -> str:
    """Blend two hex colors evenly."""
    try:
        h1, h2 = c1.lstrip("#"), c2.lstrip("#")
        r1, g1, b1 = (int(h1[i : i + 2], 16) for i in (0, 2, 4))
        r2, g2, b2 = (int(h2[i : i + 2], 16) for i in (0, 2, 4))
        return f"#{(r1 + r2)//2:02x}{(g1 + g2)//2:02x}{(b1 + b2)//2:02x}"
    except Exception:
        return "#6b7280"

# This was developed using an LLM to facilitate complex interactivity.
# JavaScript/CSS/Html is outside the scope of the course and project.
def _inject_focus_controls(html_text: str, fusions: list | None = None) -> str:
    """Add filtering controls, legend, and explanatory copy."""
    fusions_json = json.dumps(fusions or [])
    layout_styles = """
    <style id="rp-layout">
      html, body { height:100vh; overflow:hidden; }
      body { display:flex; align-items:flex-start; gap:0; margin:0; padding:0; overflow:hidden; }
      .card { flex:1 1 auto; width:100% !important; height:100vh; margin:0; padding:0; border:0; box-shadow:none; background:transparent; }
      .card-body { width:100% !important; height:100vh; padding:0 !important; }
      #mynetwork { flex: 1 1 auto; min-width:0; height:100vh; }
      #mynetwork > div.vis-network { width: 100% !important; height: 100% !important; }
      #rp-panel { order:2; display:flex; flex-direction:column; gap:10px; max-width:360px; width:340px; align-items:stretch; flex:0 0 auto; max-height:100vh; overflow-y:auto; position:sticky; top:0; }
      /* Enable scrolling for the right-side cards */
      #rp-panel::-webkit-scrollbar { width: 8px; }
      #rp-panel::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 8px; }
      #rp-panel::-webkit-scrollbar-track { background: #f8fafc; }
      .vis-tooltip { display: none !important; }
      @media (max-width: 1200px) {
        body { flex-direction: column; overflow:auto; }
        #rp-panel { order:1; width:100%; max-width:100%; position:relative; max-height:none; }
        #mynetwork { order:2; width:100%; height:70vh; }
      }
    </style>
    """
    info = """
    <div id="rp-info" style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;padding:12px 14px;box-shadow:0 12px 32px rgba(17,24,39,0.12);font-family:Arial, sans-serif;width:100%;">
      <div style="font-weight:700;font-size:18px;color:#0f172a;">Cuisine Similarity Network</div>
      <div style="margin-top:6px;font-size:13px;line-height:1.45;color:#1f2937;">Nodes are cuisines; edges show shared high-weight ingredients/features. Thickness encodes overlap strength; colors stay consistent within each region/parent.</div>
      <ul style="margin:8px 0 0 14px;padding:0;font-size:12px;line-height:1.5;color:#4b5563;">
        <li>Click a parent/region to swap it with its cuisines; click background to collapse.</li>
        <li>Use the legend to select multiple regions and see their links (even if no edges exist).</li>
        <li>Click a cuisine to isolate its neighbors; ingredient filter spotlights matches.</li>
      </ul>
      <div style="margin-top:8px;font-size:12px;line-height:1.5;color:#1f2937;">Tip: thicker, darker links mark stronger shared flavor profiles. Dual-colored links show cross-region overlap; hover nodes/edges for top shared terms.</div>
    </div>
    """
    controls = """
    <div id="rp-controls" style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;padding:12px 12px;box-shadow:0 12px 32px rgba(17,24,39,0.12);font-family:Arial, sans-serif;min-width:280px;max-width:340px;width:100%;">
      <div style="font-weight:700;margin-bottom:8px;color:#0f172a;">Filters</div>
      <div style="margin-bottom:10px;">
        <div style="font-size:12px;font-weight:600;color:#1f2937;margin-bottom:6px;">Regions / parents</div>
        <div id="rp-legend" style="display:flex;flex-wrap:wrap;gap:6px;"></div>
        <div style="margin-top:6px;display:flex;gap:8px;">
          <button id="rp-show-all" style="border:1px solid #cbd5e1;background:white;color:#1f2937;border-radius:8px;padding:6px 10px;font-size:12px;cursor:pointer;">Show parents</button>
          <button id="rp-show-children" style="border:1px solid #cbd5e1;background:white;color:#0f172a;border-radius:8px;padding:6px 10px;font-size:12px;cursor:pointer;">Show all cuisines</button>
          <button id="rp-show-reset" style="border:none;background:#111827;color:white;border-radius:8px;padding:6px 10px;font-size:12px;cursor:pointer;">Reset view</button>
        </div>
      </div>
      <div style="font-size:12px;font-weight:600;color:#1f2937;margin-bottom:6px;">Ingredient contains</div>
      <div style="display:flex;gap:6px;align-items:center;margin-bottom:6px;">
        <input id="rp-ingredient-input" type="text" placeholder="e.g., ginger" style="flex:1;border:1px solid #cbd5e1;border-radius:8px;padding:6px 8px;font-size:13px;" />
        <button id="rp-ingredient-apply" style="border:none;background:#2563eb;color:white;border-radius:8px;padding:6px 10px;font-size:13px;cursor:pointer;">Filter</button>
        <button id="rp-ingredient-clear" style="border:1px solid #cbd5e1;background:white;color:#1f2937;border-radius:8px;padding:6px 8px;font-size:13px;cursor:pointer;">Clear</button>
      </div>
      <div style="font-size:11px;color:#475569;">Legend lets you select multiple regions and see their links; ingredient filter shows matching cuisines (and their parents) even if no edges exist.</div>
    </div>
    """
    panel = f"""
    <div id="rp-panel" style="order:2; display:flex; flex-direction:column; gap:10px; max-width:360px; width:340px; align-items:stretch; flex:0 0 auto; padding-top:12px; padding-right:12px;">
      <div style="display:flex;gap:8px;align-items:center;justify-content:flex-start;margin-bottom:4px;">
        <a href="index.html" style="font-size:12px;color:#111827;font-weight:700;text-decoration:none;">Cuisine Network</a>
        <span style="color:#cbd5e1;">|</span>
        <a href="ingredient_network.html" style="font-size:12px;color:#2563eb;font-weight:700;text-decoration:none;">Ingredient Network</a>
      </div>
      {info}
      {controls}
      <div id="rp-fusion" style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;padding:12px 12px;box-shadow:0 12px 32px rgba(17,24,39,0.12);font-family:Arial, sans-serif;min-width:280px;max-width:340px;width:100%;">
        <div style="font-weight:700;margin-bottom:8px;color:#0f172a;">Fusion Lab</div>
        <div style="display:flex;flex-direction:column;gap:6px;">
          <label style="font-size:12px;color:#1f2937;font-weight:600;">Cuisine A</label>
          <select id="rp-fusion-a" style="border:1px solid #cbd5e1;border-radius:8px;padding:6px 8px;font-size:13px;"></select>
          <label style="font-size:12px;color:#1f2937;font-weight:600;">Cuisine B</label>
          <select id="rp-fusion-b" style="border:1px solid #cbd5e1;border-radius:8px;padding:6px 8px;font-size:13px;"></select>
          <button id="rp-fusion-run" style="margin-top:6px;border:none;background:#111827;color:white;border-radius:8px;padding:8px 10px;font-size:12px;cursor:pointer;">Show fusion ideas</button>
        </div>
        <div id="rp-fusion-result" style="margin-top:8px;font-size:12px;color:#334155;"></div>
      </div>
    </div>
    """
    selected_bar = """
    <div id="rp-selected" style="position:fixed;top:8px;left:8px;z-index:9998;background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;padding:8px 10px;box-shadow:0 8px 24px rgba(17,24,39,0.12);font-family:Arial,sans-serif;max-width:calc(100% - 400px);min-width:220px;">
      <div id="rp-selected-title" style="font-weight:700;color:#0f172a;font-size:14px;">Hover a node to see details</div>
      <div id="rp-selected-meta" style="font-size:12px;color:#475569;margin-top:2px;"></div>
      <div id="rp-selected-terms" style="font-size:12px;color:#1f2937;margin-top:4px;"></div>
    </div>
    """
    script = """
    <script type="text/javascript">
    (function(){
      const nodes = window.nodes;
      const edges = window.edges;
      const network = window.network;
      window.fusions = """ + fusions_json + """;
      if(!nodes || !edges || !network){ return; }
      const parentIds = new Set(nodes.get().filter(n => String(n.id).startsWith("parent::")).map(n => n.id));
      const cuisineIds = new Set(nodes.get().filter(n => !parentIds.has(n.id)).map(n => n.id));
      const childToParent = {};
      const parentToChildren = {};
      const orphanIds = new Set();
      nodes.get().forEach(function(n){
        const isParent = parentIds.has(n.id);
        const parentLabel = n.group;
        if(!isParent && parentLabel){
          const pid = "parent::" + parentLabel;
          childToParent[n.id] = pid;
          if(!parentToChildren[pid]){ parentToChildren[pid] = []; }
          parentToChildren[pid].push(n.id);
        } else if(!isParent && !parentLabel){
          orphanIds.add(n.id);
        }
      });

      let parentPositions = {};
      network.once("stabilized", function(){
        parentPositions = network.getPositions(Array.from(parentIds));
        try {
          network.fit({animation: false, padding: 40});
          network.moveTo({scale: 1.3, offset: {x:0,y:0}, animation: {duration:500, easingFunction:"easeInOutQuad"}});
        } catch(e){}
      });
      function ensureParentPositions(){
        const missing = Array.from(parentIds).filter(function(pid){ return !parentPositions[pid]; });
        if(missing.length){
          const got = network.getPositions(missing);
          Object.assign(parentPositions, got);
        }
      }

      const selection = new Set();      // parent ids selected via legend/click
      let ingredientMatches = new Set(); // cuisine/parent ids surfaced by ingredient filter

      function updateSelectedBar(data){
        const titleEl = document.getElementById("rp-selected-title");
        const metaEl = document.getElementById("rp-selected-meta");
        const termsEl = document.getElementById("rp-selected-terms");
        if(!titleEl || !metaEl || !termsEl){ return; }
        if(!data){
          titleEl.textContent = "Hover a node to see details";
          metaEl.textContent = "";
          termsEl.textContent = "";
          return;
        }
        const cleanLabel = (s) => String(s || "").replace(/parent::/gi, "").trim();
        const cleanText = (s) => cleanLabel(String(s || "")).replace(/<br>/g, ", ").replace(/\s+,/g, ",").trim();
        if(data.type === "edge"){
          const name = `${cleanLabel(data.from)} \u2194 ${cleanLabel(data.to)}`;
          titleEl.textContent = name;
          const bits = [`Shared: ${data.value || 0}`].filter(Boolean).join(" | ");
          metaEl.textContent = bits;
          termsEl.textContent = cleanText(data.title);
          return;
        }
        const name = cleanLabel(data.label || data.id);
        const parent = data.group ? `Parent: ${cleanLabel(data.group)}` : "";
        const count = data.count ? `Recipes: ${data.count}` : "";
        const degree = typeof data.value === "number" ? `Size: ${data.value.toFixed(2)}` : "";
        const bits = [parent, count, degree].filter(Boolean).join(" | ");
        titleEl.textContent = name;
        metaEl.textContent = bits;
        const terms = Array.isArray(data.terms) ? data.terms.slice(0, 6) : [];
        termsEl.textContent = terms.length ? `Top terms: ${terms.join(", ")}` : "";
      }

      function renderFusionCards(){
        const selA = document.getElementById("rp-fusion-a");
        const selB = document.getElementById("rp-fusion-b");
        const btn = document.getElementById("rp-fusion-run");
        const out = document.getElementById("rp-fusion-result");
        const fusions = Array.isArray(window.fusions) ? window.fusions : [];
        if(!selA || !selB || !btn || !out){ return; }
        const slug = (s) => String(s || "").replace(/[^A-Za-z0-9]+/g, "").toLowerCase();
        // derive cuisines from fusion pairs
        const cuisinesSet = new Set();
        fusions.forEach(f => {
          const parts = String(f.pair || "").split("+").map(p => p.trim()).filter(Boolean);
          parts.forEach(p => cuisinesSet.add(p));
        });
        const cuisines = Array.from(cuisinesSet).sort((a,b)=>String(a).localeCompare(String(b)));
        const populate = function(sel){
          sel.innerHTML = "";
          cuisines.forEach(function(c){
            const opt = document.createElement("option");
            opt.value = c;
            opt.textContent = c;
            sel.appendChild(opt);
          });
        };
        populate(selA);
        populate(selB);
        const renderFusion = function(a, b){
          out.innerHTML = "";
          if(!a || !b || a === b){
            out.textContent = "Select two different cuisines.";
            return;
          }
          const aSlug = slug(a), bSlug = slug(b);
          const key = [aSlug, bSlug].sort().join("+");
          const match = fusions.find(f => {
            const stored = String(f.__slug_pair || "").toLowerCase();
            if(stored && stored === key){ return true; }
            const parts = String(f.pair || "").split("+").map(p => slug(p)).filter(Boolean).sort().join("+");
            return parts === key;
          });
          if(!match){
            out.textContent = "No fusion data found for this pair.";
            return;
          }
          const title = document.createElement("div");
          title.style.cssText = "font-weight:700;color:#0f172a;margin-bottom:4px;";
          title.textContent = match.pair || `${a} + ${b}`;
          out.appendChild(title);
          const bridges = match.bridges || [];
          if(!bridges.length){
            out.appendChild(document.createTextNode("No recommended fusion ingredients available."));
            return;
          }
          const ul = document.createElement("ul");
          ul.style.cssText = "margin:4px 0 0 12px;padding:0;";
          bridges.slice(0,5).forEach(function(bi){
            const li = document.createElement("li");
            li.style.cssText = "list-style:disc;";
            const score = bi.score?.toFixed ? bi.score.toFixed(2) : bi.score;
            li.textContent = `${bi.name} (score ${score})`;
            ul.appendChild(li);
          });
          out.appendChild(ul);
        };
        btn.addEventListener("click", function(){
          renderFusion(selA.value, selB.value);
        });
        // initial render
        renderFusion(selA.value, selB.value);
      }

      function syncEdgeVisibility(){
        edges.forEach(function(e){
          const fromHidden = (nodes.get(e.from) || {}).hidden;
          const toHidden = (nodes.get(e.to) || {}).hidden;
          const hide = !!fromHidden || !!toHidden;
          edges.update({id: e.id, hidden: hide});
        });
      }

      function hideAll(){
        nodes.forEach(function(n){ nodes.update({id: n.id, hidden: true, physics: true}); });
      }

      function renderCollapsed(){
        nodes.forEach(function(n){
          const isParent = parentIds.has(n.id);
          const isOrphan = orphanIds.has(n.id);
          const show = isParent || isOrphan;
          nodes.update({id: n.id, hidden: !show, physics: true});
        });
        syncEdgeVisibility();
      }

      function renderSelection(){
        const hasSelection = selection.size > 0;
        const hasIngredient = ingredientMatches.size > 0;
        if(!hasSelection && !hasIngredient){
          renderCollapsed();
          return;
        }
        ensureParentPositions();
        hideAll();
        const allowed = new Set();

        // Always allow non-selected parents (collapsed)
        parentIds.forEach(function(pid){
          if(!selection.has(pid)){
            allowed.add(pid);
          }
        });

        // Track a single representative neighbor per other parent
        const largestNeighborByParent = {};

        if(hasSelection){
          // Hide the selected parents and replace with children at the parent coordinates
          selection.forEach(function(pid){
            // selected parent itself stays hidden (not added to allowed)
            const kids = parentToChildren[pid] || [];
            const pos = parentPositions[pid] || {x: 0, y: 0};
            kids.forEach(function(k, idx){
              const jitterX = (idx % 4) * 12;
              const jitterY = Math.floor(idx / 4) * 12;
              allowed.add(k);
              nodes.update({id: k, hidden: false, physics: true, x: pos.x + jitterX, y: pos.y + jitterY});
              network.getConnectedNodes(k).forEach(function(nid){
                if(parentIds.has(nid)){ return; }
                const pidNeighbor = childToParent[nid] || nid;
                if(selection.has(pidNeighbor)){ return; } // skip other selected clusters
                const node = nodes.get(nid) || {};
                const current = largestNeighborByParent[pidNeighbor];
                if(!current || (node.value || 0) > current.value){
                  largestNeighborByParent[pidNeighbor] = {id: nid, value: node.value || 0};
                }
              });
            });
          });
        }

        // Show one representative neighbor per other parent (largest by node value) and that parent
        Object.values(largestNeighborByParent).forEach(function(entry){
          if(entry && entry.id){
            allowed.add(entry.id);
            const pid = childToParent[entry.id];
            if(pid && !selection.has(pid)){
              allowed.add(pid);
            }
          }
        });

        // Ingredient overlay: keep matches and their parents visible
        if(hasIngredient){
          nodes.forEach(function(n){
            const pid = childToParent[n.id];
            const keep = ingredientMatches.has(n.id) || (pid && ingredientMatches.has(pid));
            if(keep){
              allowed.add(n.id);
              if(pid){ allowed.add(pid); }
            }
          });
        }

        // Always show orphans
        orphanIds.forEach(function(id){ allowed.add(id); });

        // Apply visibility based on allowed set
        nodes.forEach(function(n){
          const show = allowed.has(n.id);
          nodes.update({id: n.id, hidden: !show, physics: true});
        });

        syncEdgeVisibility();
      }

      function updateLegendButtons(){
        document.querySelectorAll("[data-rp-parent]").forEach(function(btn){
          const pid = btn.getAttribute("data-rp-parent");
          const active = selection.has(pid);
          btn.style.background = active ? "#111827" : "white";
          btn.style.color = active ? "white" : "#0f172a";
          btn.style.borderColor = active ? "#111827" : "#cbd5e1";
          btn.style.boxShadow = active ? "0 6px 14px rgba(15,23,42,0.24)" : "none";
        });
      }

      function toggleParent(pid){
        if(selection.has(pid)){ selection.delete(pid); } else { selection.add(pid); }
        renderSelection();
        updateLegendButtons();
      }

      function applyIngredientFilter(term){
        const t = (term || "").trim().toLowerCase();
        ingredientMatches = new Set();
        if(t){
          nodes.forEach(function(n){
            const terms = Array.isArray(n.terms) ? n.terms : [];
            if(terms.some(function(v){ return String(v).toLowerCase().includes(t); })){
              ingredientMatches.add(n.id);
            }
          });
          Array.from(ingredientMatches).forEach(function(id){
            const pid = childToParent[id];
            if(pid){ ingredientMatches.add(pid); }
          });
        }
        renderSelection();
      }

      network.on("click", function(params){
        if(params.nodes.length === 1){
          const nid = params.nodes[0];
          if(parentIds.has(nid)){
            toggleParent(nid);
            return;
          }
          if(cuisineIds.has(nid)){
            selection.clear();
            ingredientMatches = new Set([nid]);
            renderSelection();
            updateLegendButtons();
            return;
          }
        }
        if(params.nodes.length === 0){
          selection.clear();
          ingredientMatches = new Set();
          renderCollapsed();
          updateLegendButtons();
        }
      });

      network.on("hoverNode", function(params){
        if(params.node){
          const data = nodes.get(params.node);
          updateSelectedBar(data);
        }
      });
      network.on("blurNode", function(){
        updateSelectedBar(null);
      });
      network.on("hoverEdge", function(params){
        if(params.edge){
          const data = edges.get(params.edge);
          if(data){ updateSelectedBar({...data, type: "edge"}); }
        }
      });
      network.on("blurEdge", function(){
        updateSelectedBar(null);
      });

      // Legend
      const legend = document.getElementById("rp-legend");
      if(legend){
        const parents = nodes.get().filter(function(n){ return parentIds.has(n.id); }).sort(function(a,b){ return String(a.label).localeCompare(String(b.label)); });
        parents.forEach(function(p){
          const btn = document.createElement("button");
          btn.type = "button";
          btn.setAttribute("data-rp-parent", p.id);
          btn.style.cssText = "border:1px solid #cbd5e1;background:white;color:#0f172a;border-radius:8px;padding:6px 8px;font-size:12px;cursor:pointer;display:flex;align-items:center;gap:6px;";
          const swatch = document.createElement("span");
          swatch.style.cssText = "width:12px;height:12px;border-radius:6px;display:inline-block;border:1px solid #cbd5e1;background:" + (p.color || "#94a3b8");
          btn.appendChild(swatch);
          const label = document.createElement("span");
          label.textContent = p.label;
          btn.appendChild(label);
          btn.addEventListener("click", function(){ toggleParent(p.id); });
          legend.appendChild(btn);
        });
      }

      const input = document.getElementById("rp-ingredient-input");
      const applyBtn = document.getElementById("rp-ingredient-apply");
      const clearBtn = document.getElementById("rp-ingredient-clear");
      if(applyBtn && input){
        applyBtn.addEventListener("click", function(){ applyIngredientFilter(input.value); });
        input.addEventListener("keydown", function(ev){ if(ev.key === "Enter"){ applyIngredientFilter(input.value); }});
      }
      if(clearBtn){
        clearBtn.addEventListener("click", function(){
          if(input){ input.value = ""; }
          ingredientMatches = new Set();
          selection.clear();
          renderCollapsed();
          updateLegendButtons();
        });
      }
      const showAllBtn = document.getElementById("rp-show-all");
      if(showAllBtn){
        showAllBtn.addEventListener("click", function(){
          selection.clear();
          ingredientMatches = new Set();
          renderCollapsed();
          updateLegendButtons();
        });
      }
      const showChildrenBtn = document.getElementById("rp-show-children");
      if(showChildrenBtn){
        showChildrenBtn.addEventListener("click", function(){
          selection.clear();
          ingredientMatches = new Set();
          nodes.forEach(function(n){
            const isParent = parentIds.has(n.id);
            // Hide parents, show all cuisines/orphans
            nodes.update({id: n.id, hidden: isParent, physics: true});
          });
          syncEdgeVisibility();
          updateLegendButtons();
        });
      }
      const resetBtn = document.getElementById("rp-show-reset");
      if(resetBtn){
        resetBtn.addEventListener("click", function(){
          const inp = document.getElementById("rp-ingredient-input");
          if(inp){ inp.value = ""; }
          ingredientMatches = new Set();
          selection.clear();
          renderCollapsed();
          updateLegendButtons();
        });
      }

      renderCollapsed();
      renderFusionCards();
    })();
    </script>
    """
    html_text = html_text.replace("<head>", "<head>" + layout_styles)
    return html_text.replace("<body>", "<body>" + selected_bar + panel).replace("</body>", script + "</body>")

def _plot_cuisine_network_static(df_top, cuisine_counts, out_path: Path, *, parent_map=None, dedupe_map=None, min_shared=2, top_k=25):
    """Generate a simple static PNG thumbnail of the cuisine similarity network (parent/child)."""
    parent_map = parent_map or {}
    dedupe_map = dedupe_map or {}

    def _clean_label(raw: str) -> str:
        t = str(raw).strip()
        t = t.strip("[](){}\"' ")
        t = re.sub(r"[,_]+", " ", t)
        t = t.replace("-", " ").replace("_", " ")
        t = re.sub(r"\s+", " ", t)
        return t.strip()

    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", str(s).strip().lower())

    parent_lookup = {_norm(k): str(v) for k, v in parent_map.items() if str(k).strip() and str(v).strip()}
    def resolve_parent(label: str) -> str:
        if not parent_lookup:
            return label
        return parent_lookup.get(_norm(label), label)

    def _map_term(term: str) -> str | None:
        raw = str(term).strip()
        mapped = dedupe_map.get(raw.lower(), raw)
        mapped = str(mapped).strip()
        if not mapped:
            return None
        if mapped.lower() in _DROP_TOKENS or any(substr in mapped.lower() for substr in _DROP_SUBSTRINGS):
            return None
        return mapped

    # Limit cuisines and terms
    keep_cuisines = set(cuisine_counts.keys())
    df_filtered = df_top.copy()
    df_filtered["cuisine"] = df_filtered["cuisine"].map(_clean_label)
    df_filtered = df_filtered[df_filtered["cuisine"].isin(keep_cuisines)]
    if "rank" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["rank"] <= top_k]

    cuisine_to_terms = {}
    for cuisine, group in df_filtered.groupby("cuisine"):
        terms = set()
        for r in group.itertuples():
            mapped = _map_term(r.feature)
            if mapped:
                terms.add(mapped)
        cuisine_to_terms[cuisine] = terms

    cuisine_parents = {c: resolve_parent(c) for c in keep_cuisines} if parent_lookup else {}
    parent_to_children = {}
    for child, parent in cuisine_parents.items():
        parent_to_children.setdefault(parent, []).append(child)

    G = nx.Graph()
    # Add cuisine nodes
    for c in keep_cuisines:
        count = cuisine_counts.get(c, 1)
        parent = cuisine_parents.get(c)
        G.add_node(c, count=count, parent=parent, kind="cuisine")
    # Add parent nodes
    for parent, children in parent_to_children.items():
        support = sum(cuisine_counts.get(ch, 1) for ch in children)
        G.add_node(f"parent::{parent}", count=support, parent=parent, kind="parent")
        for ch in children:
            G.add_edge(f"parent::{parent}", ch, weight=0.5)

    # Add cuisine-cuisine edges based on shared terms
    cuisines = list(keep_cuisines)
    for i in range(len(cuisines)):
        for j in range(i + 1, len(cuisines)):
            c1, c2 = cuisines[i], cuisines[j]
            shared = cuisine_to_terms.get(c1, set()) & cuisine_to_terms.get(c2, set())
            if len(shared) >= min_shared:
                G.add_edge(c1, c2, weight=len(shared))

    # Colors
    parents = sorted(set(cuisine_parents.values()))
    palette = sns.color_palette("tab20", max(3, len(parents) or 3)).as_hex()
    parent_colors = {p: palette[i % len(palette)] for i, p in enumerate(parents)}
    default_color = "#6b7280"

    pos = nx.spring_layout(G, seed=42, k=0.5)
    plt.figure(figsize=(10, 9))

    # Draw cuisines
    cuisine_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "cuisine"]
    cuisine_colors = [parent_colors.get(G.nodes[n].get("parent"), default_color) for n in cuisine_nodes]
    cuisine_sizes = [np.log1p(G.nodes[n].get("count", 1)) * 180 for n in cuisine_nodes]
    nx.draw_networkx_nodes(G, pos, nodelist=cuisine_nodes, node_color=cuisine_colors, node_size=cuisine_sizes, edgecolors="white", linewidths=0.5, alpha=0.9)

    # Draw parents
    parent_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "parent"]
    if parent_nodes:
        parent_colors_list = [parent_colors.get(n.replace("parent::", ""), "#111827") for n in parent_nodes]
        parent_sizes = [np.log1p(G.nodes[n].get("count", 1)) * 220 for n in parent_nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=parent_nodes, node_color=parent_colors_list, node_size=parent_sizes, node_shape="s", edgecolors="#0f172a", linewidths=0.8, alpha=0.95)

    # Edges
    widths = [np.log1p(d.get("weight", 1)) for _, _, d in G.edges(data=True)]
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.35, edge_color="#94a3b8")

    # Labels
    labels = {n: n.replace("parent::", "") for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color="#0f172a")

    plt.title("Cuisine Similarity Network (thumbnail)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def plot_cuisine_network(df_top, cuisine_counts, out_path, *, min_shared=2, top_k=25, parent_map=None, dedupe_map=None, fusion_dir: str | Path | None = None):
    """Generate an interactive cuisine similarity graph using shared top features."""
    parent_map = parent_map or {}
    dedupe_map = dedupe_map or {}
    fusion_dir = Path(fusion_dir) if fusion_dir else Path("reports/fusion")
    fusion_payload = []
    if fusion_dir.exists():
        def _slug(name: str) -> str:
            return re.sub(r"[^A-Za-z0-9]+", "", str(name)) or "cuisine"
        for fp in sorted(fusion_dir.glob("fusion_*.json")):
            try:
                obj = json.loads(fp.read_text(encoding="utf-8"))
                obj["__file"] = fp.name
                pair = str(obj.get("pair", ""))
                parts = [p.strip() for p in pair.split("+") if str(p).strip()]
                if len(parts) == 2:
                    slugs = sorted(_slug(p) for p in parts)
                    obj["__slug_pair"] = "+".join(slugs).lower()
                fusion_payload.append(obj)
            except Exception:
                continue

    def _esc(text: str) -> str:
        return html.escape(str(text)) if text is not None else ""

    def _map_term(term: str) -> str:
        raw = str(term).strip()
        key = raw.lower()
        mapped = dedupe_map.get(key, raw)
        mapped_lower = str(mapped).lower().strip()
        if not mapped_lower:
            return None
        if mapped_lower in _DROP_TOKENS or any(substr in mapped_lower for substr in _DROP_SUBSTRINGS):
            return None
        return mapped

    def _clean_label(raw: str) -> str:
        t = str(raw).strip()
        # Drop surrounding bracket-like characters and trailing punctuation
        t = t.strip("[](){}\"' ")
        t = re.sub(r"^[\\s,;:_-]+|[\\s,;:_-]+$", "", t)
        # Normalize internal spacing
        t = re.sub(r"[_\\-]+", " ", t)
        t = re.sub(r"\\s+", " ", t)
        return t.strip()

    def _norm_label(s: str) -> str:
        t = str(s).strip()
        # Drop stray brackets/quotes and unify delimiters
        t = t.strip("[](){}\"'")
        t = re.sub(r"[,_]+", " ", t)
        t = t.replace("-", " ").replace("_", " ")
        t = re.sub(r"\s+", " ", t).lower().strip()
        return t

    parent_lookup = {_norm_label(k): str(v) for k, v in parent_map.items() if str(k).strip() and str(v).strip()}

    def resolve_parent(label: str) -> str:
        if not parent_lookup:
            return label
        return parent_lookup.get(_norm_label(label), label)

    # Keep only cuisines we care about and top-k features per cuisine
    clean_counts = {}
    for k, v in cuisine_counts.items():
        cleaned = _clean_label(k)
        clean_counts[cleaned] = clean_counts.get(cleaned, 0) + v
    cuisine_counts = clean_counts
    keep_cuisines = set(cuisine_counts.keys())
    df_filtered = df_top.copy()
    df_filtered["cuisine"] = df_filtered["cuisine"].map(_clean_label)
    df_filtered = df_filtered[df_filtered["cuisine"].isin(keep_cuisines)].copy()
    if "rank" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["rank"] <= top_k]

    # Build feature sets and weights per cuisine
    cuisine_to_terms = {}
    cuisine_term_weights = {}
    for cuisine, group in df_filtered.groupby("cuisine"):
        terms = set()
        weights = {}
        for r in group.itertuples():
            mapped = _map_term(r.feature)
            if not mapped:
                continue
            terms.add(mapped)
            weights[mapped] = weights.get(mapped, 0.0) + float(r.weight)
        cuisine_to_terms[cuisine] = terms
        cuisine_term_weights[cuisine] = weights

    cuisine_parents = {c: resolve_parent(c) for c in keep_cuisines} if parent_lookup else {}
    parent_to_children = {}
    for child, parent in cuisine_parents.items():
        parent_to_children.setdefault(parent, []).append(child)

    net = Network(height="720px", width="1080px", bgcolor="#f8f9fb", font_color="#1b1b1b", notebook=False, cdn_resources="in_line")
    G = nx.Graph()
    net.force_atlas_2based(gravity=-40, central_gravity=0.03, spring_length=180, damping=0.68, overlap=0.6)

    # Color palette for nodes (parent-aware if available). Use a fixed, high-contrast palette to avoid similar blues.
    fixed_parent_palette = [
        "#e74c3c",  # red
        "#f39c12",  # orange
        "#2ecc71",  # green
        "#1abc9c",  # teal
        "#3498db",  # blue
        "#9b59b6",  # purple
        "#e67e22",  # amber
        "#34495e",  # slate
        "#ff7f50",  # coral
        "#16a085",  # jade
        "#c0392b",  # brick
        "#8e44ad",  # violet
    ]
    fallback_palette = sns.color_palette("husl", max(12, len(keep_cuisines) or 1)).as_hex()
    if cuisine_parents:
        parents = sorted(set(cuisine_parents.values()))
        if len(parents) <= len(fixed_parent_palette):
            base_palette = fixed_parent_palette
        else:
            base_palette = sns.color_palette("tab20", max(3, len(parents))).as_hex()
        parent_colors = {p: base_palette[i % len(base_palette)] for i, p in enumerate(parents)}
        for p in parents:
            parent_colors.setdefault(p, fallback_palette[hash(p) % len(fallback_palette)])
        cuisine_colors = {}
        for c in keep_cuisines:
            parent = cuisine_parents.get(c)
            base = parent_colors.get(parent, fallback_palette[hash(parent or c) % len(fallback_palette)])
            cuisine_colors[c] = base
    else:
        base_palette = sns.color_palette("cubehelix", len(keep_cuisines) or 1).as_hex()
        parent_colors = {}
        cuisine_colors = {c: base_palette[i % len(base_palette)] for i, c in enumerate(sorted(keep_cuisines))}

    group_options = {p: {"color": parent_colors[p]} for p in parent_colors}

    # Optional parent-level feature aggregation for cross-parent edges
    parent_term_weights = {}
    parent_terms = {}
    if cuisine_parents:
        for cuisine, terms in cuisine_to_terms.items():
            parent = cuisine_parents[cuisine]
            parent_terms.setdefault(parent, set()).update(terms)
            weights = cuisine_term_weights.get(cuisine, {})
            agg = parent_term_weights.setdefault(parent, {})
            for t, w in weights.items():
                agg[t] = agg.get(t, 0.0) + w

    # Add parent nodes sized by summed support
    if cuisine_parents:
        for parent, children in sorted(parent_to_children.items()):
            support = sum(cuisine_counts.get(ch, 1) for ch in children)
            preview = ", ".join(_esc(ch) for ch in sorted(children)[:12])
            parent_node_color = parent_colors.get(parent, fallback_palette[hash(parent) % len(fallback_palette)])
            parent_colors[parent] = parent_node_color  # enforce consistency
            parent_title = f"{_esc(parent)}<br/>Recipes: {support}<br/>Children ({len(children)}): {preview}"
            net.add_node(
                f"parent::{parent}",
                label=parent,
                value=float(np.log1p(support)),
                title=parent_title,
                color=parent_node_color,
                shape="diamond",
                borderWidth=3,
                physics=True,
                font={"size": 18},
                terms=[],
                group=parent,
                count=int(support),
            )
            G.add_node(f"parent::{parent}", kind="parent", parent=parent, count=int(support))

    # Add cuisine nodes sized by support, colored by parent cluster when available
    for c in sorted(keep_cuisines):
        count = cuisine_counts.get(c, 1)
        weights = cuisine_term_weights.get(c, {})
        top_terms = sorted(weights.items(), key=lambda kv: -kv[1])[:8]
        top_terms_txt = "<br/>".join(f"{_esc(t)}: {w:.2f}" for t, w in top_terms)
        parent = cuisine_parents.get(c)
        node_color = cuisine_colors.get(c)
        if parent:
            base_parent = parent_colors.get(parent, fallback_palette[hash(parent) % len(fallback_palette)])
            parent_colors[parent] = base_parent  # ensure stored
            if not node_color:
                node_color = base_parent
        if not node_color:
            node_color = fallback_palette[hash(c) % len(fallback_palette)]
        net.add_node(
            c,
            label=c,
            value=float(np.log1p(count)),
            title=f"{_esc(c)}: {count} recipes" + (f"<br>Parent: {_esc(parent)}" if parent else "") + (f"<br>Top terms:<br>{top_terms_txt}" if top_terms else ""),
            color=node_color,
            font={"size": 14},
            terms=[t for t, _ in top_terms],
            count=int(count),
            parent=parent or "",
        )
        G.add_node(c, kind="cuisine", parent=parent or "", count=int(count))
        # Manually store group to keep color applied (pyvis ignores color when group kwarg is passed)
        if parent:
            net.nodes[-1]["group"] = parent

    # Add edges based on shared features (count + summed weight)
    cuisines = list(keep_cuisines)
    for i in range(len(cuisines)):
        for j in range(i + 1, len(cuisines)):
            c1, c2 = cuisines[i], cuisines[j]
            shared = cuisine_to_terms.get(c1, set()) & cuisine_to_terms.get(c2, set())
            if len(shared) >= min_shared:
                # Score: combined weight of shared terms for prioritizing strong overlaps
                weights1 = cuisine_term_weights.get(c1, {})
                weights2 = cuisine_term_weights.get(c2, {})
                shared_score = sum(weights1.get(t, 0) + weights2.get(t, 0) for t in shared)
                shared_preview = ", ".join(
                    _esc(s) for s in sorted(shared, key=lambda t: -(weights1.get(t, 0) + weights2.get(t, 0)))[:10]
                )
                title = f"Shared terms: {len(shared)} | Score: {shared_score:.2f}<br>{shared_preview}"
                p1, p2 = cuisine_parents.get(c1, c1), cuisine_parents.get(c2, c2)
                if p1 == p2 and p1 in parent_colors:
                    edge_color = parent_colors[p1]
                    net.add_edge(
                        c1,
                        c2,
                        value=len(shared),
                        width=max(1.5, np.log1p(len(shared)) * 1.4),
                        title=title,
                        color=edge_color,
                    )
                    G.add_edge(c1, c2, weight=len(shared))
                else:
                    width_each = max(1.0, np.log1p(len(shared)) * 0.9)
                    color1 = parent_colors.get(p1, fallback_palette[hash(p1) % len(fallback_palette)])
                    color2 = parent_colors.get(p2, fallback_palette[hash(p2) % len(fallback_palette)])
                    net.add_edge(
                        c1,
                        c2,
                        value=len(shared),
                        width=width_each,
                        title=title,
                        color=color1,
                        smooth={"type": "curvedCW", "roundness": 0.15},
                    )
                    G.add_edge(c1, c2, weight=len(shared))
                    net.add_edge(
                        c2,
                        c1,
                        value=len(shared),
                        width=width_each,
                        title=title,
                        color=color2,
                        smooth={"type": "curvedCCW", "roundness": 0.15},
                    )

    # Connect parents to children for interactive cluster exploration
    if cuisine_parents:
        for child, parent in cuisine_parents.items():
            net.add_edge(
                f"parent::{parent}",
                child,
                value=1,
                width=1,
                color=parent_colors.get(parent, "#555"),
                title=f"{_esc(child)} belongs to {_esc(parent)}",
                dashes=True,
            )
            G.add_edge(f"parent::{parent}", child, weight=1)

        # Optional: parent-to-parent edges showing overlap of child feature spaces
        parent_list = list(parent_terms.keys())
        for i in range(len(parent_list)):
            for j in range(i + 1, len(parent_list)):
                p1, p2 = parent_list[i], parent_list[j]
                shared = parent_terms[p1] & parent_terms[p2]
                if len(shared) >= max(2, min_shared):
                    w1 = parent_term_weights.get(p1, {})
                    w2 = parent_term_weights.get(p2, {})
                    shared_score = sum(w1.get(t, 0) + w2.get(t, 0) for t in shared)
                    shared_preview = ", ".join(_esc(s) for s in sorted(shared, key=lambda t: -(w1.get(t, 0) + w2.get(t, 0)))[:8])
                    net.add_edge(
                        f"parent::{p1}",
                        f"parent::{p2}",
                        value=len(shared),
                        width=max(1.5, np.log1p(len(shared)) * 1.1),
                        title=f"Parent overlap: {len(shared)} shared terms<br>{shared_preview}<br>Score: {shared_score:.2f}",
                        color=_blend_colors(
                            parent_colors.get(p1, fallback_palette[hash(p1) % len(fallback_palette)]),
                            parent_colors.get(p2, fallback_palette[hash(p2) % len(fallback_palette)]),
                        ),
                        dashes=True,
                    )
                    G.add_edge(f"parent::{p1}", f"parent::{p2}", weight=len(shared))

    net.set_options(
        json.dumps(
            {
                "nodes": {
                    "shape": "dot",
                    "scaling": {"min": 8, "max": 36},
                    "shadow": True,
                    "borderWidth": 1,
                    "borderWidthSelected": 3,
                    "font": {"size": 16, "strokeWidth": 0, "face": "arial"},
                },
                "edges": {
                    "smooth": {"type": "dynamic"},
                    "color": {"inherit": False},
                    "shadow": False,
                    "selectionWidth": 2,
                },
                "configure": {"enabled": False},
                "layout": {"improvedLayout": True},
                "physics": {"stabilization": {"iterations": 150}, "timestep": 0.35, "minVelocity": 0.75},
                "interaction": {
                    "hover": True,
                    "hoverConnectedEdges": True,
                    "selectConnectedEdges": True,
                    "multiselect": False,
                    "tooltipDelay": 100,
                    "navigationButtons": False,
                    "hideEdgesOnDrag": False,
                    "zoomView": True,
                    "dragView": True,
                    "keyboard": False,
                },
                "groups": group_options,
            }
        )
    )
    # Generate HTML and write with UTF-8 to avoid Windows default cp1252 errors
    net.conf = False
    net.html = _inject_focus_controls(net.generate_html(notebook=False), fusion_payload)
    Path(out_path).write_text(net.html, encoding="utf-8")
    # Save basic node metrics (degree and betweenness)
    bet = nx.betweenness_centrality(G, normalized=True)
    rows = []
    for node in G.nodes():
        rows.append({
            "node": node,
            "degree": G.degree(node),
            "betweenness": bet.get(node, 0.0),
            "parent": G.nodes[node].get("parent", ""),
            "kind": G.nodes[node].get("kind", "cuisine"),
        })
    metrics_path = Path(out_path).with_name("cuisine_network_metrics.csv")
    pd.DataFrame(rows).to_csv(metrics_path, index=False)

def run(context: PipelineContext, *, force: bool = False) -> StageResult:
    cfg = context.stage("analysis_graph")
    logger = stage_logger(context, "analysis_graph", force=force)
    
    # 1. Config & Inputs
    baseline_stage_cfg = context.stage("analysis_baseline")
    recommender_cfg = context.stage("analysis_recommender")
    baseline_output_cfg = baseline_stage_cfg.get("output", {})
    reports_dir = Path(baseline_output_cfg.get("reports_dir", "reports/baseline"))
    
    top_feat_path = reports_dir / "top_features_logreg.csv"
    viz_dir = Path(cfg.get("output", {}).get("viz_dir", "reports/viz_graph"))
    viz_dir.mkdir(parents=True, exist_ok=True)

    parent_map_path = baseline_stage_cfg.get("params", {}).get("parent_map_path")
    parent_map = {}
    if parent_map_path:
        p = Path(parent_map_path)
        if p.exists():
            try:
                parent_map = json.load(open(p, "r", encoding="utf-8"))
            except Exception as exc:  # pragma: no cover - safety net
                logger.warning("Failed to load parent map %s: %s", p, exc)
        else:
            logger.warning("Parent map not found at %s", p)
    
    # Optional ingredient dedupe map for cleaning displayed terms
    dedupe_map = {}
    dedupe_map_path = Path(cfg.get("data", {}).get("dedupe_map_path", "data/ingr_normalized/dedupe_map.jsonl"))
    if dedupe_map_path.exists():
        try:
            dedupe_map = load_jsonl_map(dedupe_map_path)
            logger.info("Loaded %d dedupe terms from %s", len(dedupe_map), dedupe_map_path)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to load dedupe map %s: %s", dedupe_map_path, exc)
    else:
        logger.info("Dedupe map not found at %s; proceeding without ingredient remap", dedupe_map_path)
    
    if not top_feat_path.exists():
        logger.warning(f"Missing {top_feat_path}. Run analysis_baseline first.")
        return StageResult(name="analysis_graph", status="skipped", details="Missing input data")
        
    # 2. Load Data
    df_top = pd.read_csv(top_feat_path)
    
    # Infer frequency from predictions (preferred) or top_feat file (fallback).
    preds_path = reports_dir / "y_pred_logreg.csv"
    cuisine_counts_series = None
    if preds_path.exists():
        df_preds = pd.read_csv(preds_path)
        if "y_true" in df_preds.columns:
            cuisine_counts_series = df_preds["y_true"].value_counts()
        elif "y_true_parent" in df_preds.columns:
            # Fallback to parent-level counts if only parent labels exist
            cuisine_counts_series = df_preds["y_true_parent"].value_counts()
    if cuisine_counts_series is None:
        cuisine_counts_series = df_top["cuisine"].value_counts()
    # Keep top N children for readability
    max_nodes = int(cfg.get("params", {}).get("max_nodes", 30))
    cuisine_counts_series = cuisine_counts_series.head(max_nodes)
    cuisine_counts = cuisine_counts_series.to_dict()
    
    # 3. Generate Visualizations
    
    # [cite_start]A. Interactive Network [cite: 330]
    html_path = viz_dir / "cuisine_network.html"
    logger.info(f"Generating network graph -> {html_path}")
    import numpy as np # Ensure numpy is available for log1p
    fusion_dir = Path(recommender_cfg.get("output", {}).get("reports_dir", "reports/fusion"))
    plot_cuisine_network(df_top, cuisine_counts, html_path, min_shared=2, parent_map=parent_map, dedupe_map=dedupe_map, fusion_dir=fusion_dir)
    # Also copy to public for static hosting
    try:
        public_dir = Path("public")
        public_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(html_path, public_dir / "cuisine_network.html")
        shutil.copy(html_path, public_dir / "index.html")
    except Exception as exc:
        logger.warning("Failed to copy cuisine network to public: %s", exc)
    # Static thumbnail for reports
    thumb_path = viz_dir / "cuisine_network.png"
    try:
        _plot_cuisine_network_static(df_top, cuisine_counts, thumb_path, parent_map=parent_map, dedupe_map=dedupe_map, min_shared=2, top_k=25)
    except Exception as exc:
        logger.warning("Failed to generate cuisine network thumbnail: %s", exc)
    
    # [cite_start]B. Distribution Plot [cite: 307]
    dist_path = viz_dir / "cuisine_distribution.png"
    plt.figure(figsize=(10, 10))
    counts_df = cuisine_counts_series.reset_index()
    counts_df.columns = ["cuisine", "count"]
    sns.barplot(data=counts_df, x="count", y="cuisine", palette="viridis", dodge=False, hue="cuisine", legend=False)
    plt.title("Cuisine Distribution (Top by support)")
    plt.xlabel("Frequency")
    plt.ylabel("Cuisine")
    plt.tight_layout()
    plt.savefig(dist_path)
    plt.close()
    
    return StageResult(
        name="analysis_graph", 
        status="success", 
        outputs={"network": str(html_path), "plot": str(dist_path)}
    )
