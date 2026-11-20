-- Pandoc Lua filter used by scripts/build_paper.py.
--
-- * Pipe tables -> booktabs `tabular` inside `table` (< 5 columns, \small)
--   or `table*` (>= 5 columns, \footnotesize), wrapped in \adjustbox so a
--   wide table is shrunk to the column/text width instead of overflowing.
--   The paragraph immediately before a table that starts with "**Table N.**"
--   is used as the caption (label stripped; LaTeX prints "TABLE N").
-- * Images -> `figure` (width=1col, default) or `figure*` (width=2col) with
--   \includegraphics, \caption and \label. A missing image file becomes a
--   framed placeholder box of about the right size.
--
-- Metadata expected: root_dir (repository root, for checking that an image
-- file exists). Figures are referenced as figures/<basename>; the Python
-- driver copies the files into paper/build/figures/.

local root_dir = "."
local WIDE_COLS = 5

local writer_opts = pandoc.WriterOptions { cite_method = "natbib", wrap_text = "wrap-none" }

local function trim(s)
  return (s:gsub("^%s+", ""):gsub("%s+$", ""))
end

local function blocks_latex(blocks)
  return trim(pandoc.write(pandoc.Pandoc(blocks), "latex", writer_opts))
end

local function inlines_latex(inlines)
  return blocks_latex({ pandoc.Plain(inlines) })
end

local function basename(path)
  return (path:gsub("^.*/", ""))
end

local function file_exists(path)
  local f = io.open(path, "rb")
  if f then f:close() return true end
  return false
end

local function align_char(a)
  if a == "AlignCenter" then return "c" end
  if a == "AlignRight" then return "r" end
  return "l"
end

-- If `para` starts with **Table N.** return (caption inlines, N), else nil.
local function table_caption(para)
  if para.t ~= "Para" or #para.content == 0 then return nil end
  local first = para.content[1]
  if first.t ~= "Strong" then return nil end
  local label = trim(pandoc.utils.stringify(first))
  local num = label:match("^Table%s+(%d+)%.?$")
  if not num then return nil end
  local rest = {}
  for i = 2, #para.content do rest[#rest + 1] = para.content[i] end
  while #rest > 0 and (rest[1].t == "Space" or rest[1].t == "SoftBreak") do
    table.remove(rest, 1)
  end
  return rest, num
end

local function row_latex(row)
  local cells = {}
  for _, c in ipairs(row.cells) do cells[#cells + 1] = blocks_latex(c.contents) end
  return table.concat(cells, " & ") .. " \\\\"
end

local function table_to_latex(tbl, caption_inlines, num)
  local ncol = #tbl.colspecs
  local spec = {}
  for _, cs in ipairs(tbl.colspecs) do spec[#spec + 1] = align_char(cs[1]) end
  -- a table is set across both columns when it has WIDE_COLS or more columns, or when its header
  -- (joined) is at least 40 characters: short tables with long text cells would otherwise be shrunk
  -- by adjustbox to an unreadable size inside one column
  local header_len = 0
  for _, row in ipairs(tbl.head.rows) do
    for _, c in ipairs(row.cells) do header_len = header_len + #pandoc.utils.stringify(c.contents) end
  end
  local wide = ncol >= WIDE_COLS or header_len >= 40
  local env = wide and "table*" or "table"
  local size = wide and "\\footnotesize" or "\\small"
  local maxw = wide and "\\textwidth" or "\\columnwidth"

  local caption
  if caption_inlines then
    caption = inlines_latex(caption_inlines)
  elseif tbl.caption and tbl.caption.long and #tbl.caption.long > 0 then
    caption = blocks_latex(tbl.caption.long)
  end

  local out = {}
  out[#out + 1] = "\\begin{" .. env .. "}[t]"
  out[#out + 1] = "\\centering" .. size
  if caption then
    out[#out + 1] = "\\caption{" .. caption .. "}"
    if num then out[#out + 1] = "\\label{tab:" .. num .. "}" end
  end
  out[#out + 1] = "\\adjustbox{max width=" .. maxw .. "}{%"
  out[#out + 1] = "\\begin{tabular}{@{}" .. table.concat(spec) .. "@{}}"
  out[#out + 1] = "\\toprule"
  for _, row in ipairs(tbl.head.rows) do out[#out + 1] = row_latex(row) end
  out[#out + 1] = "\\midrule"
  for _, body in ipairs(tbl.bodies) do
    for _, row in ipairs(body.head) do out[#out + 1] = row_latex(row) end
    for _, row in ipairs(body.body) do out[#out + 1] = row_latex(row) end
  end
  for _, row in ipairs(tbl.foot.rows) do out[#out + 1] = row_latex(row) end
  out[#out + 1] = "\\bottomrule"
  out[#out + 1] = "\\end{tabular}}"
  out[#out + 1] = "\\end{" .. env .. "}"
  return pandoc.RawBlock("latex", table.concat(out, "\n"))
end

local function image_to_latex(img, caption_inlines, identifier)
  local width = img.attributes["width"] or "1col"
  local wide = (width == "2col")
  local env = wide and "figure*" or "figure"
  local w = wide and "\\textwidth" or "\\columnwidth"
  local height = img.attributes["height"] or (wide and "2.6in" or "2.0in")
  local id = identifier
  if id == nil or id == "" then id = img.identifier end
  local src = img.src
  local abs = src
  if not src:match("^/") then abs = root_dir .. "/" .. src end

  local body
  if file_exists(abs) then
    body = "\\includegraphics[width=" .. w .. "]{figures/" .. basename(src) .. "}"
  else
    body = "\\fbox{\\parbox[c][" .. height .. "][c]{\\dimexpr " .. w ..
      "-2\\fboxsep-2\\fboxrule\\relax}{\\centering\\small [Figure placeholder: \\texttt{" ..
      src:gsub("_", "\\_") .. "} not found]}}"
  end
  local out = {}
  out[#out + 1] = "\\begin{" .. env .. "}[t]"
  out[#out + 1] = "\\centering"
  out[#out + 1] = body
  if caption_inlines and #caption_inlines > 0 then
    out[#out + 1] = "\\caption{" .. inlines_latex(caption_inlines) .. "}"
  end
  if id and id ~= "" then out[#out + 1] = "\\label{" .. id .. "}" end
  out[#out + 1] = "\\end{" .. env .. "}"
  return pandoc.RawBlock("latex", table.concat(out, "\n"))
end

local function figure_to_latex(fig)
  local img
  fig:walk { Image = function(im) if not img then img = im end end }
  if not img then return fig end
  local cap = pandoc.utils.blocks_to_inlines(fig.caption.long or {})
  return image_to_latex(img, cap, fig.identifier)
end

function Pandoc(doc)
  if doc.meta.root_dir then root_dir = pandoc.utils.stringify(doc.meta.root_dir) end
  local blocks = doc.blocks
  local out = {}
  local i = 1
  while i <= #blocks do
    local b = blocks[i]
    local nxt = blocks[i + 1]
    local handled = false
    if b.t == "Para" and nxt and nxt.t == "Table" then
      local cap, num = table_caption(b)
      if cap then
        out[#out + 1] = table_to_latex(nxt, cap, num)
        i = i + 2
        handled = true
      end
    end
    if not handled then
      if b.t == "Table" then
        out[#out + 1] = table_to_latex(b, nil, nil)
      elseif b.t == "Figure" then
        out[#out + 1] = figure_to_latex(b)
      elseif b.t == "Para" and #b.content == 1 and b.content[1].t == "Image" then
        local img = b.content[1]
        out[#out + 1] = image_to_latex(img, img.caption, img.identifier)
      else
        out[#out + 1] = b
      end
      i = i + 1
    end
  end
  doc.blocks = out
  return doc
end
