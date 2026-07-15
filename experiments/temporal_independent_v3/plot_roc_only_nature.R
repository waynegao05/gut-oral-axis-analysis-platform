suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(jsonlite)
  library(patchwork)
  library(scales)
  library(tibble)
})

BASE_FONT <- "Times New Roman"
NEUTRAL_DARK <- "#30343B"
CHANCE <- "#B7A3A3"
SPLIT_42 <- "#3F7392"
SPLIT_43 <- "#B57932"
WHITE <- "#FFFFFF"

parse_args <- function(args) {
  values <- list(
    input_dir = "outputs/current_mainline_v2/temporal_independent_v3/figure_diagnostics",
    output_dir = "outputs/evidence_report/figures",
    source_dir = "outputs/evidence_report/source_data_nature"
  )
  index <- 1
  while (index <= length(args)) {
    key <- args[[index]]
    if (key == "--input-dir" && index < length(args)) {
      values$input_dir <- args[[index + 1]]
      index <- index + 2
    } else if (key == "--output-dir" && index < length(args)) {
      values$output_dir <- args[[index + 1]]
      index <- index + 2
    } else if (key == "--source-dir" && index < length(args)) {
      values$source_dir <- args[[index + 1]]
      index <- index + 2
    } else {
      stop(sprintf("Unknown or incomplete argument: %s", key))
    }
  }
  values
}

theme_nature_roc <- function(base_size = 6.6) {
  theme_classic(base_size = base_size, base_family = BASE_FONT) +
    theme(
      axis.line = element_blank(),
      axis.ticks = element_line(linewidth = 0.34, colour = NEUTRAL_DARK),
      axis.ticks.length = grid::unit(1.7, "pt"),
      axis.title = element_text(size = base_size, colour = NEUTRAL_DARK),
      axis.text = element_text(size = base_size - 0.35, colour = NEUTRAL_DARK),
      panel.border = element_rect(fill = NA, colour = NEUTRAL_DARK, linewidth = 0.36),
      panel.grid = element_blank(),
      plot.title = element_text(
        size = base_size + 0.45,
        face = "bold",
        colour = NEUTRAL_DARK,
        margin = margin(l = 9, b = 2.5)
      ),
      plot.tag = element_text(size = 8.1, face = "bold", colour = NEUTRAL_DARK),
      plot.tag.position = c(0, 1),
      plot.margin = margin(4, 6, 4, 5, unit = "pt"),
      legend.title = element_blank(),
      legend.text = element_text(size = 6.3),
      legend.key.width = grid::unit(13, "pt"),
      legend.key.height = grid::unit(5, "pt"),
      legend.spacing.x = grid::unit(4, "pt"),
      legend.margin = margin(0, 0, 0, 0)
    )
}

read_roc_report <- function(path, split_label) {
  if (!file.exists(path)) {
    stop(sprintf("Required ROC report does not exist: %s", path))
  }
  report <- fromJSON(path, simplifyVector = FALSE)
  bind_rows(lapply(report$horizons, function(row) {
    tibble(
      split = split_label,
      horizon = as.numeric(row$horizon),
      auc = as.numeric(row$auc),
      num_cases = as.integer(row$num_cases),
      num_controls = as.integer(row$num_controls),
      num_excluded_censored = as.integer(row$num_excluded_censored),
      fpr = as.numeric(unlist(row$false_positive_rate)),
      tpr = as.numeric(unlist(row$true_positive_rate))
    )
  }))
}

write_source_csv <- function(data, path) {
  connection <- file(path, open = "wt", encoding = "UTF-8")
  on.exit(close(connection), add = TRUE)
  write.csv(data, connection, row.names = FALSE, na = "")
}

make_roc_panel <- function(data, horizon_value, tag, show_x, show_y, show_legend) {
  panel_data <- filter(data, horizon == horizon_value)
  auc_rows <- panel_data %>% distinct(split, auc)
  auc42 <- auc_rows$auc[auc_rows$split == "Split 42"]
  auc43 <- auc_rows$auc[auc_rows$split == "Split 43"]

  ggplot(panel_data, aes(x = fpr, y = tpr, colour = split)) +
    geom_abline(
      intercept = 0,
      slope = 1,
      linewidth = 0.38,
      linetype = "44",
      colour = CHANCE
    ) +
    geom_step(linewidth = 0.64, direction = "hv") +
    scale_colour_manual(
      values = c("Split 42" = SPLIT_42, "Split 43" = SPLIT_43),
      breaks = c("Split 42", "Split 43")
    ) +
    scale_x_continuous(
      breaks = seq(0, 1, 0.25),
      labels = c("0", "0.25", "0.50", "0.75", "1"),
      limits = c(0, 1),
      expand = c(0, 0)
    ) +
    scale_y_continuous(
      breaks = seq(0, 1, 0.25),
      labels = c("0", "0.25", "0.50", "0.75", "1"),
      limits = c(0, 1),
      expand = c(0, 0)
    ) +
    coord_equal() +
    labs(
      tag = tag,
      title = sprintf("t = %.0f | AUC %.3f / %.3f", horizon_value, auc42, auc43),
      x = if (show_x) "False-positive rate" else NULL,
      y = if (show_y) "True-positive rate" else NULL,
      colour = NULL
    ) +
    theme_nature_roc() +
    theme(legend.position = if (show_legend) "top" else "none")
}

save_figure <- function(plot, output_base, width_mm = 183, height_mm = 66) {
  width_in <- width_mm / 25.4
  height_in <- height_mm / 25.4

  svglite::svglite(
    paste0(output_base, ".svg"),
    width = width_in,
    height = height_in,
    bg = WHITE
  )
  print(plot)
  grDevices::dev.off()

  grDevices::cairo_pdf(
    paste0(output_base, ".pdf"),
    width = width_in,
    height = height_in,
    family = BASE_FONT,
    onefile = TRUE,
    bg = WHITE
  )
  print(plot)
  grDevices::dev.off()

  ragg::agg_tiff(
    paste0(output_base, ".tiff"),
    width = width_in,
    height = height_in,
    units = "in",
    res = 600,
    background = WHITE,
    compression = "lzw"
  )
  print(plot)
  grDevices::dev.off()

  ragg::agg_png(
    paste0(output_base, ".png"),
    width = width_in,
    height = height_in,
    units = "in",
    res = 300,
    background = WHITE
  )
  print(plot)
  grDevices::dev.off()
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  font_match <- systemfonts::match_fonts(BASE_FONT)
  if (nrow(font_match) == 0 || !file.exists(font_match$path[[1]])) {
    stop(sprintf("Required font is not available: %s", BASE_FONT))
  }

  dir.create(args$output_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(args$source_dir, recursive = TRUE, showWarnings = FALSE)

  roc_data <- bind_rows(
    read_roc_report(file.path(args$input_dir, "split42_consensus_roc.json"), "Split 42"),
    read_roc_report(file.path(args$input_dir, "split43_consensus_roc.json"), "Split 43")
  ) %>%
    mutate(split = factor(split, levels = c("Split 42", "Split 43")))
  horizons <- sort(unique(roc_data$horizon))
  if (!identical(horizons, c(36, 60, 84))) {
    stop("Expected ROC horizons 36, 60, and 84.")
  }
  if (any(!is.finite(roc_data$fpr)) || any(!is.finite(roc_data$tpr))) {
    stop("ROC source contains non-finite curve coordinates.")
  }

  auc_summary <- roc_data %>%
    distinct(split, horizon, auc, num_cases, num_controls, num_excluded_censored)
  write_source_csv(
    roc_data,
    file.path(args$source_dir, "fig27_temporal_topology_roc_curve_points.csv")
  )
  write_source_csv(
    auc_summary,
    file.path(args$source_dir, "fig27_temporal_topology_roc_auc_summary.csv")
  )

  panel_a <- make_roc_panel(roc_data, horizons[[1]], "a", FALSE, TRUE, TRUE)
  panel_b <- make_roc_panel(roc_data, horizons[[2]], "b", TRUE, FALSE, FALSE)
  panel_c <- make_roc_panel(roc_data, horizons[[3]], "c", FALSE, FALSE, FALSE)

  figure <- (panel_a | panel_b | panel_c) +
    plot_layout(guides = "collect") +
    plot_annotation(
      theme = theme(
        text = element_text(family = BASE_FONT, colour = NEUTRAL_DARK),
        plot.margin = margin(4, 5, 4, 5, unit = "pt"),
        legend.position = "top",
        legend.justification = "center"
      )
    )

  output_base <- file.path(args$output_dir, "fig27_temporal_topology_roc_nature_v3")
  save_figure(figure, output_base)

  qa_notes <- c(
    "Figure contract",
    "Core conclusion: The final temporal-topology consensus shows reproducible time-dependent discrimination in two test splits.",
    "Visual backend: R only (ggplot2 + patchwork; svglite/cairo_pdf/ragg export).",
    "Font: Times New Roman for all text.",
    "Metric: cumulative/dynamic IPCW ROC at t = 36, 60, and 84.",
    "Model: final validation-only cross-split consensus alpha = 0.63.",
    "Displayed text is restricted to panel labels, horizon/AUC labels, legend, and shared axes.",
    "AUC pair order follows the legend: Split 42 / Split 43.",
    "Dataset limitation: topology_v6 is synthetic/noisy augmented and is not an external clinical validation cohort.",
    "Source data: source_data_nature/fig27_temporal_topology_roc_*.csv."
  )
  writeLines(
    qa_notes,
    file.path(dirname(args$output_dir), "fig27_temporal_topology_roc_nature_v3_qa.txt"),
    useBytes = TRUE
  )

  message(sprintf("Saved ROC-only Nature figure to %s", normalizePath(args$output_dir, winslash = "/")))
}

main()
